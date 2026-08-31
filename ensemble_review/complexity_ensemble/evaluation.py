from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .ensemble import RoutedOutput
from .nde import DampedPendulum, MorseRoutedVectorField, integrate_rk4
from .pinn import HeatPINN, heat_solution


@dataclass
class PINNMetrics:
    relative_l2: float
    mse: float
    mae: float
    max_absolute_error: float
    physics_residual_mse: float
    interface_mse: float
    mean_complex_weight: float
    hard_complex_fraction: float
    sequence_alignment_mse: float


@dataclass
class PINNEvaluation:
    metrics: PINNMetrics
    t_grid: torch.Tensor
    x_grid: torch.Tensor
    prediction: torch.Tensor
    reference: torch.Tensor
    absolute_error: torch.Tensor
    complexity: torch.Tensor
    complex_weight: torch.Tensor


@dataclass
class NDEMetrics:
    trajectory_relative_l2: float
    trajectory_rmse: float
    trajectory_mae: float
    final_state_l2: float
    vector_field_mse: float
    kinematic_residual_mse: float
    interface_mse: float
    mean_complex_weight: float
    hard_complex_fraction: float


@dataclass
class NDEEvaluation:
    metrics: NDEMetrics
    times: torch.Tensor
    prediction: torch.Tensor
    reference: torch.Tensor
    absolute_error: torch.Tensor
    complexity: torch.Tensor
    complex_weight: torch.Tensor


def _relative_l2(prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(prediction - reference) / torch.linalg.vector_norm(reference).clamp_min(1e-12)


def evaluate_heat_pinn(model: HeatPINN, n_time: int = 101, n_space: int = 101) -> PINNEvaluation:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    t = torch.linspace(0.0, 1.0, n_time, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, n_space, device=device, dtype=dtype)
    t_grid, x_grid = torch.meshgrid(t, x, indexing="ij")
    points = torch.stack((t_grid.flatten(), x_grid.flatten()), dim=-1)

    model.eval()
    with torch.no_grad():
        routed = model(points, return_details=True)
        assert isinstance(routed, RoutedOutput)
        prediction = routed.value
        reference = heat_solution(points, model.diffusivity)
        error = (prediction - reference).abs()
        complexity = model.router.complexity(points)
        hard_fraction = model.router.complex_weight(points, hard=True).mean()
        interface = model.ensemble.interface_loss(points)
        alignment_sum = torch.zeros((), device=device, dtype=dtype)
        for chunk in points.split(1024):
            alignment_sum = alignment_sum + model.sequence_alignment_loss(chunk) * len(chunk)
        alignment = alignment_sum / len(points)
    # Residual requires first and second input derivatives, even during evaluation.
    with torch.enable_grad():
        residual_sum = torch.zeros((), device=device, dtype=dtype)
        for chunk in points.split(1024):
            residual_sum = residual_sum + model.residual(chunk).square().sum().detach()
        residual_mse = residual_sum / len(points)

    metrics = PINNMetrics(
        relative_l2=float(_relative_l2(prediction, reference)),
        mse=float((prediction - reference).square().mean()),
        mae=float(error.mean()),
        max_absolute_error=float(error.max()),
        physics_residual_mse=float(residual_mse),
        interface_mse=float(interface),
        mean_complex_weight=float(routed.complex_weight.mean()),
        hard_complex_fraction=float(hard_fraction),
        sequence_alignment_mse=float(alignment),
    )
    shape = (n_time, n_space)
    return PINNEvaluation(
        metrics,
        t_grid.detach().cpu(),
        x_grid.detach().cpu(),
        prediction.reshape(shape).detach().cpu(),
        reference.reshape(shape).detach().cpu(),
        error.reshape(shape).detach().cpu(),
        complexity.reshape(shape).detach().cpu(),
        routed.complex_weight.reshape(shape).detach().cpu(),
    )


def evaluate_pendulum_nde(
    model: MorseRoutedVectorField,
    initial_state: torch.Tensor | None = None,
    final_time: float = 20.0,
    n_steps: int = 401,
) -> NDEEvaluation:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if initial_state is None:
        initial_state = torch.tensor([1.2, 0.0], device=device, dtype=dtype)
    else:
        initial_state = initial_state.to(device=device, dtype=dtype)
    times = torch.linspace(0.0, final_time, n_steps, device=device, dtype=dtype)
    true_field = DampedPendulum()

    model.eval()
    with torch.no_grad():
        prediction = integrate_rk4(model, initial_state, times)
        reference = integrate_rk4(true_field, initial_state, times)
        inputs = torch.cat((times[:, None], prediction), dim=-1)
        routed = model.ensemble(inputs, return_details=True)
        assert isinstance(routed, RoutedOutput)
        true_derivative = true_field(times, prediction)
        error = (prediction - reference).abs()
        vector_field_mse = (routed.value - true_derivative).square().mean()
        kinematic_mse = (routed.value[:, 0] - prediction[:, 1]).square().mean()
        interface = model.ensemble.interface_loss(inputs)
        complexity = model.router.complexity(inputs)
        hard_fraction = model.router.complex_weight(inputs, hard=True).mean()

    metrics = NDEMetrics(
        trajectory_relative_l2=float(_relative_l2(prediction, reference)),
        trajectory_rmse=float((prediction - reference).square().mean().sqrt()),
        trajectory_mae=float(error.mean()),
        final_state_l2=float(torch.linalg.vector_norm(prediction[-1] - reference[-1])),
        vector_field_mse=float(vector_field_mse),
        kinematic_residual_mse=float(kinematic_mse),
        interface_mse=float(interface),
        mean_complex_weight=float(routed.complex_weight.mean()),
        hard_complex_fraction=float(hard_fraction),
    )
    return NDEEvaluation(
        metrics,
        times.detach().cpu(),
        prediction.detach().cpu(),
        reference.detach().cpu(),
        error.detach().cpu(),
        complexity.detach().cpu(),
        routed.complex_weight.detach().cpu(),
    )


def save_metrics(metrics: PINNMetrics | NDEMetrics, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metrics), indent=2) + "\n", encoding="utf-8")


def format_metrics(metrics: PINNMetrics | NDEMetrics) -> str:
    return "\n".join(f"{name:28s}: {value:.6e}" for name, value in asdict(metrics).items())
