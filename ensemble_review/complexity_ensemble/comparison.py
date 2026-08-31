from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev

import torch
from torch import nn

from .baselines import SingleComplexHeatPINN, SingleComplexVectorField
from .nde import DampedPendulum, MorseRoutedVectorField, integrate_rk4, sample_pendulum_derivatives
from .pinn import HeatPINN, heat_solution, sample_heat_problem


@dataclass
class ComparisonRecord:
    problem: str
    model: str
    simple_expert: str
    complex_expert: str
    seed: int
    training_noise: float
    inference_noise: float
    clean_error: float
    ood_error: float
    physics_error: float
    parameters: int
    mean_complex_weight: float


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _relative_l2(prediction: torch.Tensor, reference: torch.Tensor) -> float:
    value = torch.linalg.vector_norm(prediction - reference) / torch.linalg.vector_norm(reference).clamp_min(1e-12)
    return float(value)


def _heat_points(device: torch.device, *, ood: bool, n_points: int = 2601) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(982451653 if ood else 7)
    start, width = (1.0, 0.25) if ood else (0.0, 1.0)
    time = start + width * torch.rand(n_points, 1, generator=generator, device=device)
    space = 2.0 * torch.rand(n_points, 1, generator=generator, device=device) - 1.0
    return torch.cat((time, space), dim=-1)


def _heat_errors(
    model: nn.Module,
    clean_points: torch.Tensor,
    inference_noise: float,
    noise_seed: int,
) -> tuple[float, float, float]:
    generator = torch.Generator(device=clean_points.device).manual_seed(noise_seed)
    observed_points = clean_points + inference_noise * torch.randn(
        clean_points.shape, generator=generator, device=clean_points.device
    )
    with torch.no_grad():
        error = _relative_l2(model(observed_points), heat_solution(clean_points))
        usage = (
            float(model.router.complex_weight(observed_points).mean())
            if isinstance(model, HeatPINN)
            else 1.0
        )
    physics = float(model.residual(observed_points).square().mean().detach())
    return error, physics, usage


def _heat_ood_error(
    model: nn.Module,
    device: torch.device,
    inference_noise: float,
    noise_seed: int,
    n_points: int = 2048,
) -> float:
    generator = torch.Generator(device=device).manual_seed(982451653)
    time = 1.0 + 0.25 * torch.rand(n_points, 1, generator=generator, device=device)
    space = 2.0 * torch.rand(n_points, 1, generator=generator, device=device) - 1.0
    points = torch.cat((time, space), dim=-1)
    noise_generator = torch.Generator(device=device).manual_seed(noise_seed)
    observed_points = points + inference_noise * torch.randn(
        points.shape, generator=noise_generator, device=device
    )
    with torch.no_grad():
        return _relative_l2(model(observed_points), heat_solution(points))


def _train_heat_model(
    model: nn.Module,
    data: tuple[torch.Tensor, ...],
    epochs: int,
    learning_rate: float,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    initial_tx, initial_u, boundary_tx, boundary_u, collocation = data
    for _ in range(epochs):
        optimizer.zero_grad()
        if isinstance(model, HeatPINN):
            loss = model.losses(*data).total
        elif isinstance(model, SingleComplexHeatPINN):
            loss = model.training_loss(data)
        else:
            initial_loss = (model(initial_tx) - initial_u).square().mean()
            boundary_loss = (model(boundary_tx) - boundary_u).square().mean()
            physics_loss = model.residual(collocation).square().mean()
            loss = initial_loss + boundary_loss + physics_loss
        loss.backward()
        optimizer.step()


def compare_heat_models(
    simple_kind: str,
    complex_kind: str = "mlp",
    seeds: tuple[int, ...] = (0, 1, 2),
    noise_levels: tuple[float, ...] = (0.0, 0.05),
    inference_noise_levels: tuple[float, ...] = (0.0, 0.05),
    epochs: int = 500,
    learning_rate: float = 1e-3,
    n_initial: int = 128,
    n_boundary: int = 128,
    n_collocation: int = 1024,
    device: torch.device | str = "cpu",
) -> list[ComparisonRecord]:
    device = torch.device(device)
    records: list[ComparisonRecord] = []
    for seed in seeds:
        torch.manual_seed(seed)
        clean_data = sample_heat_problem(n_initial, n_boundary, n_collocation, device=device)
        reference_points = torch.cat((clean_data[0], clean_data[2], clean_data[4]))
        for noise in noise_levels:
            generator = torch.Generator(device=device).manual_seed(seed + 100003)
            data = list(clean_data)
            data[1] = data[1] + noise * torch.randn(data[1].shape, generator=generator, device=device)
            data[3] = data[3] + noise * torch.randn(data[3].shape, generator=generator, device=device)
            for model_name in ("morse", "learned", "single_complex"):
                torch.manual_seed(seed)
                if model_name == "single_complex":
                    model: nn.Module = SingleComplexHeatPINN(complex_kind=complex_kind).to(device)
                else:
                    model = HeatPINN(
                        simple_kind=simple_kind,
                        complex_kind=complex_kind,
                        routing=model_name,
                    ).to(device)
                    model.fit_router(reference_points)
                _train_heat_model(model, tuple(data), epochs, learning_rate)
                evaluation_points = _heat_points(device, ood=False)
                for inference_noise in inference_noise_levels:
                    clean_error, physics_error, usage = _heat_errors(
                        model, evaluation_points, inference_noise, seed + 200003
                    )
                    records.append(
                        ComparisonRecord(
                            "heat",
                            model_name,
                            simple_kind if model_name != "single_complex" else "none",
                            complex_kind,
                            seed,
                            noise,
                            inference_noise,
                            clean_error,
                            _heat_ood_error(
                                model, device, inference_noise, seed + 300007
                            ),
                            physics_error,
                            parameter_count(model),
                            usage,
                        )
                    )
    return records


def _train_nde_model(
    model: nn.Module,
    inputs: torch.Tensor,
    target: torch.Tensor,
    epochs: int,
    learning_rate: float,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        optimizer.zero_grad()
        if isinstance(model, MorseRoutedVectorField):
            loss = model.losses(inputs, target).total
        else:
            prediction = model(inputs[:, 0], inputs[:, 1:])
            data_loss = (prediction - target).square().mean()
            kinematic = (prediction[:, 0] - inputs[:, 2]).square().mean()
            loss = data_loss + 0.1 * kinematic
        loss.backward()
        optimizer.step()


class _NoisyObservationField(nn.Module):
    def __init__(self, model: nn.Module, noise: float, seed: int) -> None:
        super().__init__()
        self.model = model
        self.noise = noise
        self.generator: torch.Generator | None = None
        self.seed = seed

    def forward(self, t: torch.Tensor | float, state: torch.Tensor) -> torch.Tensor:
        if self.generator is None:
            self.generator = torch.Generator(device=state.device).manual_seed(self.seed)
        observed = state + self.noise * torch.randn(
            state.shape, generator=self.generator, device=state.device, dtype=state.dtype
        )
        return self.model(t, observed)


def _nde_errors(
    model: nn.Module,
    initial_state: torch.Tensor,
    inference_noise: float,
    noise_seed: int,
    final_time: float = 10.0,
    n_steps: int = 201,
) -> tuple[float, float, float]:
    times = torch.linspace(0.0, final_time, n_steps, device=initial_state.device)
    true_field = DampedPendulum()
    observed_field = _NoisyObservationField(model, inference_noise, noise_seed)
    with torch.no_grad():
        prediction = integrate_rk4(observed_field, initial_state, times)
        reference = integrate_rk4(true_field, initial_state, times)
        generator = torch.Generator(device=initial_state.device).manual_seed(noise_seed + 1)
        observed_states = prediction + inference_noise * torch.randn(
            prediction.shape, generator=generator, device=prediction.device
        )
        field_error = (model(times, observed_states) - true_field(times, prediction)).square().mean()
        if isinstance(model, MorseRoutedVectorField):
            routed_inputs = torch.cat((times[:, None], observed_states), dim=-1)
            usage = float(model.router.complex_weight(routed_inputs).mean())
        else:
            usage = 1.0
    return _relative_l2(prediction, reference), float(field_error), usage


def compare_nde_models(
    simple_kind: str,
    seeds: tuple[int, ...] = (0, 1, 2),
    noise_levels: tuple[float, ...] = (0.0, 0.05),
    inference_noise_levels: tuple[float, ...] = (0.0, 0.05),
    epochs: int = 500,
    learning_rate: float = 1e-3,
    n_samples: int = 2048,
    device: torch.device | str = "cpu",
) -> list[ComparisonRecord]:
    device = torch.device(device)
    records: list[ComparisonRecord] = []
    for seed in seeds:
        torch.manual_seed(seed)
        inputs, clean_target = sample_pendulum_derivatives(n_samples, device=device)
        for noise in noise_levels:
            generator = torch.Generator(device=device).manual_seed(seed + 100003)
            target = clean_target + noise * torch.randn(
                clean_target.shape, generator=generator, device=device
            )
            for model_name in ("morse", "learned", "single_complex"):
                torch.manual_seed(seed)
                if model_name == "single_complex":
                    model: nn.Module = SingleComplexVectorField().to(device)
                else:
                    model = MorseRoutedVectorField(2, simple_kind=simple_kind, routing=model_name).to(device)
                    model.router.fit(inputs)
                _train_nde_model(model, inputs, target, epochs, learning_rate)
                initial = torch.tensor([1.2, 0.0], device=device)
                ood_initial = torch.tensor([2.5, 0.5], device=device)
                for inference_noise in inference_noise_levels:
                    clean_error, physics_error, usage = _nde_errors(
                        model, initial, inference_noise, seed + 200003
                    )
                    ood_error, _, _ = _nde_errors(
                        model, ood_initial, inference_noise, seed + 300007
                    )
                    records.append(
                        ComparisonRecord(
                            "pendulum",
                            model_name,
                            simple_kind if model_name != "single_complex" else "none",
                            "mlp",
                            seed,
                            noise,
                            inference_noise,
                            clean_error,
                            ood_error,
                            physics_error,
                            parameter_count(model),
                            usage,
                        )
                    )
    return records


def save_comparison(records: list[ComparisonRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def comparison_summary(records: list[ComparisonRecord]) -> str:
    baseline_training_noise = min(record.training_noise for record in records)
    baseline_inference_noise = min(record.inference_noise for record in records)
    baseline_by_model = {
        model_name: mean(
            record.clean_error
            for record in records
            if record.model == model_name
            and record.training_noise == baseline_training_noise
            and record.inference_noise == baseline_inference_noise
        )
        for model_name in ("morse", "learned", "single_complex")
    }
    complex_experts = ", ".join(sorted({record.complex_expert for record in records}))
    lines = [
        f"complex expert: {complex_experts}",
        "model            train N  infer N  clean error (mean +/- std)    OOD error (mean +/- std)    degradation",
    ]
    for model_name in ("morse", "learned", "single_complex"):
        for training_noise in sorted({record.training_noise for record in records}):
            for inference_noise in sorted({record.inference_noise for record in records}):
                subset = [
                    r
                    for r in records
                    if r.model == model_name
                    and r.training_noise == training_noise
                    and r.inference_noise == inference_noise
                ]
                clean = [r.clean_error for r in subset]
                ood = [r.ood_error for r in subset]
                clean_std = stdev(clean) if len(clean) > 1 else 0.0
                ood_std = stdev(ood) if len(ood) > 1 else 0.0
                degradation = 100.0 * (mean(clean) / baseline_by_model[model_name] - 1.0)
                lines.append(
                    f"{model_name:16s} {training_noise:7.3f} {inference_noise:7.3f}  "
                    f"{mean(clean):.4e} +/- {clean_std:.2e}        "
                    f"{mean(ood):.4e} +/- {ood_std:.2e}       {degradation:+7.2f}%"
                )
    return "\n".join(lines)


def plot_comparison(records: list[ComparisonRecord], path: str | Path) -> None:
    from .visualization import _finish_figure, _pyplot

    plt = _pyplot()
    models = ("morse", "learned", "single_complex")
    training_noises = sorted({record.training_noise for record in records})
    inference_noises = sorted({record.inference_noise for record in records})
    clean_train, noisy_train = training_noises[0], training_noises[-1]
    clean_infer, noisy_infer = inference_noises[0], inference_noises[-1]
    panels = (
        ("clean_error", clean_train, clean_infer, "Clean train / clean inference"),
        ("clean_error", noisy_train, clean_infer, f"Training noise={noisy_train:g}"),
        ("clean_error", clean_train, noisy_infer, f"Inference noise={noisy_infer:g}"),
        ("ood_error", clean_train, noisy_infer, "OOD with noisy inference"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    for axis, (field, training_noise, inference_noise, title) in zip(axes.flat[:4], panels):
        means, errors = [], []
        for model_name in models:
            values = [
                getattr(record, field)
                for record in records
                if record.model == model_name
                and record.training_noise == training_noise
                and record.inference_noise == inference_noise
            ]
            means.append(mean(values))
            errors.append(stdev(values) if len(values) > 1 else 0.0)
        axis.bar(models, means, yerr=errors, capsize=4, color=("tab:blue", "tab:orange", "tab:green"))
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_ylabel("lower is better")
        axis.grid(axis="y", alpha=0.25)

    for axis, changed_axis, title in (
        (axes.flat[4], "training", "Degradation from training noise"),
        (axes.flat[5], "inference", "Degradation from inference noise"),
    ):
        degradations = []
        for model_name in models:
            baseline_values = [
                record.clean_error
                for record in records
                if record.model == model_name
                and record.training_noise == clean_train
                and record.inference_noise == clean_infer
            ]
            changed_values = [
                record.clean_error
                for record in records
                if record.model == model_name
                and record.training_noise == (noisy_train if changed_axis == "training" else clean_train)
                and record.inference_noise == (noisy_infer if changed_axis == "inference" else clean_infer)
            ]
            degradations.append(100.0 * (mean(changed_values) / mean(baseline_values) - 1.0))
        axis.bar(models, degradations, color=("tab:blue", "tab:orange", "tab:green"))
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_ylabel("percent change; lower is better")
        axis.grid(axis="y", alpha=0.25)
    complex_experts = ", ".join(sorted({record.complex_expert for record in records}))
    figure.suptitle(
        f"{records[0].problem}: {complex_experts} routing accuracy and robustness",
        fontsize=15,
    )
    _finish_figure(figure, path)
