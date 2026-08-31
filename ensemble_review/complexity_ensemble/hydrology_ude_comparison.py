from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev

import torch
from torch import nn

from .hydrology_ude import HydrologyUDEData, RoutedHydrologyUDE, SingleComplexHydrologyUDE


@dataclass(frozen=True)
class HydrologyUDERecord:
    country: str
    basin: str
    model: str
    simple_expert: str
    complex_expert: str
    seed: int
    validation_nse: float
    test_nse: float
    test_kge: float
    test_rmse_mm_day: float
    mass_balance_rmse: float
    negative_state_fraction: float
    parameters: int
    mean_complex_weight: float


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float]:
    prediction = prediction.flatten().double()
    target = target.flatten().double()
    error = prediction - target
    rmse = torch.sqrt(error.square().mean())
    nse = 1.0 - error.square().sum() / (target - target.mean()).square().sum().clamp_min(1e-12)
    prediction_std = prediction.std(unbiased=False)
    target_std = target.std(unbiased=False)
    correlation = (
        ((prediction - prediction.mean()) * (target - target.mean())).mean()
        / (prediction_std * target_std).clamp_min(1e-12)
    )
    alpha = prediction_std / target_std.clamp_min(1e-12)
    beta = prediction.mean() / target.mean().clamp_min(1e-12)
    kge = 1.0 - torch.sqrt((correlation - 1).square() + (alpha - 1).square() + (beta - 1).square())
    return float(nse), float(kge), float(rmse)


def train_hydrology_ude(
    model: RoutedHydrologyUDE | SingleComplexHydrologyUDE,
    data: HydrologyUDEData,
    *,
    epochs: int = 20,
    chunk_days: int = 128,
    warmup_days: int = 30,
    learning_rate: float = 1e-3,
    interface_weight: float = 0.01,
    routing_weight: float = 0.02,
    state_weight: float = 0.1,
    device: torch.device | str = "cpu",
) -> None:
    if epochs < 1 or chunk_days < 2 or warmup_days < 0:
        raise ValueError("epochs must be positive, chunk_days >= 2, and warmup_days nonnegative")
    device = torch.device(device)
    forcing = data.forcings[: data.train_end].to(device)
    target = data.discharge[: data.train_end].to(device)
    target_scale = target.std(unbiased=False).clamp_min(1e-6)
    if isinstance(model, RoutedHydrologyUDE):
        model.fit_router(forcing)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        state = model.initial_state()
        for start in range(0, len(forcing), chunk_days):
            stop = min(start + chunk_days, len(forcing))
            result = model.rollout(forcing[start:stop], state)
            local_warmup = min(warmup_days, len(result.discharge) - 1) if start == 0 else 0
            prediction = result.discharge[local_warmup:]
            observed = target[start + local_warmup : stop]
            scaled_mse = ((prediction - observed) / target_scale).square().mean()
            log_mse = (torch.log1p(prediction.clamp_min(0.0)) - torch.log1p(observed)).square().mean()
            state_penalty = result.states[local_warmup:].clamp_max(0.0).square().mean()
            interface = result.interface_disagreement[local_warmup:].mean()
            if isinstance(model, RoutedHydrologyUDE) and model.routing_kind == "learned":
                usage = result.complex_weight.mean()
                routing = (usage - model.router.target_complex_fraction).square()
                routing = routing + 0.01 * (
                    result.complex_weight * (1.0 - result.complex_weight)
                ).mean()
            else:
                routing = torch.zeros((), device=device)
            loss = (
                scaled_mse + 0.2 * log_mse + state_weight * state_penalty
                + interface_weight * interface + routing_weight * routing
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            state = result.states[-1:].detach()


@torch.no_grad()
def evaluate_hydrology_ude(
    model: RoutedHydrologyUDE | SingleComplexHydrologyUDE,
    data: HydrologyUDEData,
    *,
    device: torch.device | str = "cpu",
) -> tuple[tuple[float, float, float], tuple[float, float, float], float, float, float]:
    device = torch.device(device)
    result = model.rollout(data.forcings.to(device))
    target = data.discharge.to(device)
    validation = _metrics(
        result.discharge[data.train_end : data.validation_end],
        target[data.train_end : data.validation_end],
    )
    test = _metrics(result.discharge[data.validation_end :], target[data.validation_end :])
    test_mass = result.mass_balance_residual[data.validation_end :]
    test_states = result.states[data.validation_end :]
    mass_rmse = float(torch.sqrt(test_mass.square().mean()))
    negative_fraction = float((test_states < 0.0).float().mean())
    usage = (
        float(result.complex_weight[data.validation_end :].mean())
        if isinstance(model, RoutedHydrologyUDE) else 1.0
    )
    return validation, test, mass_rmse, negative_fraction, usage


def compare_hydrology_ude_models(
    data: HydrologyUDEData,
    *,
    simple_kind: str = "rbf",
    seeds: tuple[int, ...] = (0,),
    epochs: int = 20,
    chunk_days: int = 128,
    learning_rate: float = 1e-3,
    device: torch.device | str = "cpu",
) -> list[HydrologyUDERecord]:
    device = torch.device(device)
    records = []
    for seed in seeds:
        for model_name in ("morse", "learned", "single_complex"):
            torch.manual_seed(seed)
            if model_name == "single_complex":
                model: RoutedHydrologyUDE | SingleComplexHydrologyUDE = SingleComplexHydrologyUDE(
                    data.forcing_names, data.forcing_mean, data.forcing_scale
                ).to(device)
            else:
                model = RoutedHydrologyUDE(
                    data.forcing_names, data.forcing_mean, data.forcing_scale,
                    simple_kind=simple_kind, routing=model_name,
                ).to(device)
            train_hydrology_ude(
                model, data, epochs=epochs, chunk_days=chunk_days,
                learning_rate=learning_rate, device=device,
            )
            validation, test, mass, negative, usage = evaluate_hydrology_ude(model, data, device=device)
            records.append(
                HydrologyUDERecord(
                    data.country, data.basin_id, model_name,
                    simple_kind if model_name != "single_complex" else "none",
                    "mlp", seed, validation[0], test[0], test[1], test[2],
                    mass, negative,
                    sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                    usage,
                )
            )
    return records


def save_hydrology_ude_comparison(records: list[HydrologyUDERecord], path: str | Path) -> None:
    if not records:
        raise ValueError("cannot save empty records")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def hydrology_ude_summary(records: list[HydrologyUDERecord]) -> str:
    lines = [
        f"{records[0].country}, basin {records[0].basin}; UDE complex expert: MLP",
        "model             test NSE (mean +/- std)   KGE       RMSE mm/day  mass RMSE    complex use",
    ]
    for model_name in ("morse", "learned", "single_complex"):
        subset = [record for record in records if record.model == model_name]
        nse = [record.test_nse for record in subset]
        spread = stdev(nse) if len(nse) > 1 else 0.0
        lines.append(
            f"{model_name:17s} {mean(nse):8.4f} +/- {spread:.4f}  "
            f"{mean(record.test_kge for record in subset):8.4f}  "
            f"{mean(record.test_rmse_mm_day for record in subset):11.4f}  "
            f"{mean(record.mass_balance_rmse for record in subset):10.2e}  "
            f"{mean(record.mean_complex_weight for record in subset):10.3f}"
        )
    return "\n".join(lines)


def plot_hydrology_ude_comparison(records: list[HydrologyUDERecord], path: str | Path) -> None:
    from .visualization import _finish_figure, _pyplot

    plt = _pyplot()
    models = ("morse", "learned", "single_complex")
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, field, title in (
        (axes[0], "test_nse", "Test NSE"),
        (axes[1], "test_rmse_mm_day", "Test RMSE (mm/day)"),
        (axes[2], "mean_complex_weight", "Complex-expert usage"),
    ):
        groups = [[getattr(record, field) for record in records if record.model == model] for model in models]
        axis.bar(models, [mean(group) for group in groups], yerr=[stdev(group) if len(group) > 1 else 0 for group in groups], capsize=4)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=15)
    figure.suptitle(f"Hydrology UDE: {records[0].country} basin {records[0].basin}")
    _finish_figure(figure, path)
