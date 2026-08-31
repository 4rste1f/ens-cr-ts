from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev

import torch
from torch import nn

from .hydrology import (
    LearnedHydrologyMorsePotential,
    RoutedHydrologyModel,
    SingleComplexHydrologyModel,
    SingleSimpleHydrologyModel,
)
from .hydrology_data import HydrologyData, HydrologySplit


DYNAMIC_NOISE_FEATURES = {"precipitation", "temperature", "pet", "previous_discharge"}


@dataclass(frozen=True)
class HydrologyComparisonRecord:
    country: str
    basin: str
    model: str
    simple_expert: str
    complex_expert: str
    seed: int
    training_noise: float
    inference_noise: float
    validation_nse: float
    test_nse: float
    test_kge: float
    test_rmse_mm_day: float
    physics_error: float
    parameters: int
    mean_complex_weight: float


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _add_observation_noise(
    inputs: torch.Tensor,
    feature_names: tuple[str, ...],
    noise: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Perturb observed dynamic inputs in training-standardized units."""
    if noise < 0.0:
        raise ValueError("noise levels must be non-negative")
    if noise == 0.0:
        return inputs
    mask = torch.tensor(
        [name in DYNAMIC_NOISE_FEATURES for name in feature_names],
        device=inputs.device,
        dtype=inputs.dtype,
    )[None, None, :]
    return inputs + noise * torch.randn(
        inputs.shape, generator=generator, device=inputs.device
    ) * mask


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float]:
    prediction = prediction.flatten().double()
    target = target.flatten().double()
    error = prediction - target
    rmse = torch.sqrt(error.square().mean())
    denominator = (target - target.mean()).square().sum().clamp_min(1e-12)
    nse = 1.0 - error.square().sum() / denominator
    prediction_std = prediction.std(unbiased=False)
    target_std = target.std(unbiased=False)
    covariance = ((prediction - prediction.mean()) * (target - target.mean())).mean()
    correlation = covariance / (prediction_std * target_std).clamp_min(1e-12)
    alpha = prediction_std / target_std.clamp_min(1e-12)
    beta = prediction.mean() / target.mean().clamp_min(1e-12)
    kge = 1.0 - torch.sqrt((correlation - 1.0).square() + (alpha - 1.0).square() + (beta - 1.0).square())
    return float(nse), float(kge), float(rmse)


def _make_model(
    model_name: str,
    data: HydrologyData,
    simple_kind: str,
    complex_kind: str,
    learned_morse_potential: LearnedHydrologyMorsePotential | None = None,
) -> RoutedHydrologyModel | SingleComplexHydrologyModel:
    sequence_length = data.train.inputs.shape[1]
    if model_name == "single_complex":
        return SingleComplexHydrologyModel(
            data.feature_names, sequence_length, data.discharge_scale,
            complex_kind=complex_kind,
        )
    return RoutedHydrologyModel(
        data.feature_names, sequence_length, data.discharge_scale,
        simple_kind=simple_kind, complex_kind=complex_kind, routing=model_name,
        learned_morse_potential=learned_morse_potential,
    )


def train_hydrology_model(
    model: RoutedHydrologyModel | SingleComplexHydrologyModel | SingleSimpleHydrologyModel,
    split: HydrologySplit,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    training_noise: float,
    seed: int,
    device: torch.device,
    feature_names: tuple[str, ...] | None = None,
) -> None:
    if training_noise < 0.0:
        raise ValueError("training_noise must be non-negative")
    if training_noise and feature_names is None:
        raise ValueError("feature_names are required when training_noise is non-zero")
    inputs = split.inputs.to(device)
    physical = split.physical_inputs.to(device)
    targets = split.targets.to(device)
    if isinstance(model, RoutedHydrologyModel):
        model.fit_router(inputs)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device=device).manual_seed(seed + 104729)
    for _ in range(epochs):
        permutation = torch.randperm(len(inputs), generator=generator, device=device)
        for start in range(0, len(inputs), batch_size):
            indices = permutation[start : start + batch_size]
            batch_inputs = _add_observation_noise(
                inputs[indices], feature_names or (), training_noise, generator
            )
            optimizer.zero_grad()
            losses = model.losses(batch_inputs, physical[indices], targets[indices])
            losses.total.backward()
            optimizer.step()


@torch.no_grad()
def _evaluate(
    model: RoutedHydrologyModel | SingleComplexHydrologyModel,
    split: HydrologySplit,
    data: HydrologyData,
    *,
    inference_noise: float,
    seed: int,
    device: torch.device,
) -> tuple[float, float, float, float, float]:
    inputs = split.inputs.to(device)
    physical = split.physical_inputs.to(device)
    if inference_noise:
        generator = torch.Generator(device=device).manual_seed(seed)
        inputs = _add_observation_noise(inputs, data.feature_names, inference_noise, generator)
    prediction_scaled = model(inputs)
    prediction = prediction_scaled * data.discharge_scale.to(device)
    target = split.targets.to(device) * data.discharge_scale.to(device)
    nse, kge, rmse = _metrics(prediction, target)
    physics = float(model.physics_residual(prediction_scaled, physical).square().mean())
    if isinstance(model, RoutedHydrologyModel):
        usage = float(model.router.complex_weight(inputs.flatten(start_dim=1)).mean())
    else:
        usage = 1.0
    return nse, kge, rmse, physics, usage


def compare_hydrology_models(
    data: HydrologyData,
    *,
    models: tuple[str, ...] | None = None,
    simple_kind: str = "rbf",
    complex_kind: str = "mlp",
    seeds: tuple[int, ...] = (0,),
    training_noise_levels: tuple[float, ...] = (0.0,),
    inference_noise_levels: tuple[float, ...] = (0.0, 0.05),
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: torch.device | str = "cpu",
    learned_morse_potential: LearnedHydrologyMorsePotential | None = None,
) -> list[HydrologyComparisonRecord]:
    """Run Morse, learned-gate, and architecture-matched unitary baselines."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if any(noise < 0.0 for noise in (*training_noise_levels, *inference_noise_levels)):
        raise ValueError("noise levels must be non-negative")
    available_models = {"morse", "learned", "single_complex"}
    if learned_morse_potential is not None:
        available_models.add("learned_morse")
    if models is None:
        selected_models = ["morse"]
        if learned_morse_potential is not None:
            selected_models.append("learned_morse")
        selected_models.extend(("learned", "single_complex"))
    else:
        selected_models = list(dict.fromkeys(models))
        invalid_models = set(selected_models) - available_models
        if invalid_models:
            choices = ", ".join(sorted(available_models))
            invalid = ", ".join(sorted(invalid_models))
            raise ValueError(f"unknown or unavailable models: {invalid}; choose from {choices}")
        if not selected_models:
            raise ValueError("models must not be empty")
    device = torch.device(device)
    records = []
    for seed in seeds:
        for training_noise in sorted(set(training_noise_levels)):
            for model_name in selected_models:
                torch.manual_seed(seed)
                model = _make_model(
                    model_name, data, simple_kind, complex_kind, learned_morse_potential
                ).to(device)
                train_hydrology_model(
                    model, data.train, epochs=epochs, batch_size=batch_size,
                    learning_rate=learning_rate, training_noise=training_noise,
                    seed=seed, device=device, feature_names=data.feature_names,
                )
                for inference_noise in sorted(set(inference_noise_levels)):
                    validation_nse, _, _, _, _ = _evaluate(
                        model, data.validation, data, inference_noise=inference_noise,
                        seed=seed + 200003, device=device,
                    )
                    test_nse, test_kge, test_rmse, physics, usage = _evaluate(
                        model, data.test, data, inference_noise=inference_noise,
                        seed=seed + 300007, device=device,
                    )
                    records.append(
                        HydrologyComparisonRecord(
                            data.country, data.basin_id, model_name,
                            simple_kind if model_name != "single_complex" else "none",
                            complex_kind, seed, training_noise, inference_noise,
                            validation_nse, test_nse, test_kge, test_rmse,
                            physics, _parameter_count(model), usage,
                        )
                    )
    return records


def save_hydrology_comparison(
    records: list[HydrologyComparisonRecord], path: str | Path
) -> None:
    if not records:
        raise ValueError("cannot save an empty comparison")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def hydrology_comparison_summary(records: list[HydrologyComparisonRecord]) -> str:
    if not records:
        raise ValueError("cannot summarize an empty comparison")
    lines = [
        f"{records[0].country}, basin {records[0].basin}; complex expert: {records[0].complex_expert}",
        "model             train N  infer N  test NSE (mean +/- std)   KGE       RMSE mm/day  physics",
    ]
    model_order = ("morse", "learned_morse", "learned", "single_complex")
    for model_name in model_order:
        for training_noise in sorted({record.training_noise for record in records}):
            for inference_noise in sorted({record.inference_noise for record in records}):
                subset = [
                    record for record in records
                    if record.model == model_name
                    and record.training_noise == training_noise
                    and record.inference_noise == inference_noise
                ]
                if not subset:
                    continue
                nse = [record.test_nse for record in subset]
                spread = stdev(nse) if len(nse) > 1 else 0.0
                lines.append(
                    f"{model_name:17s} {training_noise:7.3f} {inference_noise:7.3f}  "
                    f"{mean(nse):8.4f} +/- {spread:.4f}  "
                    f"{mean(record.test_kge for record in subset):8.4f}  "
                    f"{mean(record.test_rmse_mm_day for record in subset):11.4f}  "
                    f"{mean(record.physics_error for record in subset):8.4f}"
                )
    return "\n".join(lines)


def plot_hydrology_comparison(
    records: list[HydrologyComparisonRecord], path: str | Path
) -> None:
    from .visualization import _finish_figure, _pyplot

    model_order = ("morse", "learned_morse", "learned", "single_complex")
    models = tuple(model for model in model_order if any(record.model == model for record in records))
    conditions = sorted(
        {(record.training_noise, record.inference_noise) for record in records}
    )
    plt = _pyplot()
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fields = (
        ("test_nse", "Test NSE", False),
        ("test_rmse_mm_day", "Test RMSE (mm/day)", True),
        ("physics_error", "Reservoir residual MSE", True),
    )
    for axis, (field, title, lower_better) in zip(axes, fields):
        width = 0.8 / len(conditions)
        all_averages = []
        for condition_index, (training_noise, inference_noise) in enumerate(conditions):
            values = [
                [
                    getattr(record, field)
                    for record in records
                    if record.model == model
                    and record.training_noise == training_noise
                    and record.inference_noise == inference_noise
                ]
                for model in models
            ]
            averages = [mean(group) for group in values]
            errors = [stdev(group) if len(group) > 1 else 0.0 for group in values]
            positions = [
                index + (condition_index - (len(conditions) - 1) / 2.0) * width
                for index in range(len(models))
            ]
            axis.bar(
                positions,
                averages,
                width=width,
                yerr=errors,
                capsize=3,
                label=f"train {training_noise:g}, infer {inference_noise:g}",
            )
            all_averages.extend(averages)
        axis.set_xticks(range(len(models)), models, rotation=15)
        if lower_better and all(value > 0 for value in all_averages):
            axis.set_yscale("log")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[-1].legend(fontsize=8, title="Noise (standardized)")
    figure.suptitle(f"{records[0].country} basin {records[0].basin}: {records[0].complex_expert}")
    _finish_figure(figure, path)
