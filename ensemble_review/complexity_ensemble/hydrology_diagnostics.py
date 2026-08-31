from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .hydrology import HydrologyRoutedOutput, RoutedHydrologyModel
from .hydrology_data import HydrologyData, HydrologySplit


@dataclass(frozen=True)
class RoutingDiagnostic:
    split: str
    samples: int
    pearson_simple_error: float
    spearman_simple_error: float
    pearson_complex_advantage: float
    spearman_complex_advantage: float
    pearson_physics_advantage: float
    spearman_physics_advantage: float
    pearson_discharge: float
    pearson_absolute_flow_change: float
    top_fraction: float
    top_advantage_precision: float
    complex_better_high_score: float
    complex_better_low_score: float
    mean_advantage_high_score: float
    mean_advantage_low_score: float


@dataclass(frozen=True)
class RoutingSamples:
    dates: tuple[str, ...]
    score: torch.Tensor
    weight: torch.Tensor
    target: torch.Tensor
    simple_prediction: torch.Tensor
    complex_prediction: torch.Tensor
    simple_error: torch.Tensor
    complex_advantage: torch.Tensor
    physics_advantage: torch.Tensor
    absolute_flow_change: torch.Tensor
    high_flow: torch.Tensor


def _pearson(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.flatten().double()
    second = second.flatten().double()
    first = first - first.mean()
    second = second - second.mean()
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if float(denominator) <= 1e-12:
        return float("nan")
    return float((first * second).sum() / denominator)


def _ranks(values: torch.Tensor) -> torch.Tensor:
    """Return zero-based ranks; hydrology scores/errors are effectively continuous."""
    order = torch.argsort(values.flatten())
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(len(order), device=values.device, dtype=torch.float64)
    return ranks


def _spearman(first: torch.Tensor, second: torch.Tensor) -> float:
    return _pearson(_ranks(first), _ranks(second))


@torch.no_grad()
def collect_routing_samples(
    model: RoutedHydrologyModel,
    split: HydrologySplit,
    data: HydrologyData,
    *,
    device: torch.device | str = "cpu",
) -> RoutingSamples:
    device = torch.device(device)
    inputs = split.inputs.to(device)
    physical = split.physical_inputs.to(device)
    target = split.targets.to(device) * data.discharge_scale.to(device)
    routed = model(inputs, return_details=True)
    assert isinstance(routed, HydrologyRoutedOutput)
    scale = data.discharge_scale.to(device)
    simple = F.softplus(routed.simple_raw).squeeze(-1) * scale
    complex_prediction = F.softplus(routed.complex_raw).squeeze(-1) * scale
    simple_error = (simple - target).square()
    complex_error = (complex_prediction - target).square()
    complex_advantage = simple_error - complex_error
    simple_physics = model.physics_residual((simple / scale)[:, None], physical).square()
    complex_physics = model.physics_residual(
        (complex_prediction / scale)[:, None], physical
    ).square()
    previous_flow = physical[:, -1, data.feature_names.index("previous_discharge")]
    training_high_flow = torch.quantile(
        data.train.targets.to(device) * scale, 0.9
    )
    flat = inputs.flatten(start_dim=1)
    return RoutingSamples(
        tuple(day.isoformat() for day in split.target_dates),
        model.router.complexity(flat).cpu(),
        routed.complex_weight.cpu(),
        target.cpu(),
        simple.cpu(),
        complex_prediction.cpu(),
        simple_error.cpu(),
        complex_advantage.cpu(),
        (simple_physics - complex_physics).cpu(),
        (target - previous_flow).abs().cpu(),
        (target >= training_high_flow).to(torch.float32).cpu(),
    )


def routing_diagnostic(
    samples: RoutingSamples,
    *,
    split: str,
    top_fraction: float = 0.2,
) -> RoutingDiagnostic:
    if not 0.0 < top_fraction < 1.0:
        raise ValueError("top_fraction must be strictly between zero and one")
    count = len(samples.score)
    if count < 3:
        raise ValueError("at least three samples are required")
    top_count = max(1, math.ceil(top_fraction * count))
    score_indices = torch.topk(samples.score, top_count).indices
    advantage_indices = torch.topk(samples.complex_advantage, top_count).indices
    score_mask = torch.zeros(count, dtype=torch.bool)
    advantage_mask = torch.zeros(count, dtype=torch.bool)
    score_mask[score_indices] = True
    advantage_mask[advantage_indices] = True
    top_precision = float((score_mask & advantage_mask).sum() / top_count)
    complex_better = samples.complex_advantage > 0.0
    return RoutingDiagnostic(
        split,
        count,
        _pearson(samples.score, samples.simple_error),
        _spearman(samples.score, samples.simple_error),
        _pearson(samples.score, samples.complex_advantage),
        _spearman(samples.score, samples.complex_advantage),
        _pearson(samples.score, samples.physics_advantage),
        _spearman(samples.score, samples.physics_advantage),
        _pearson(samples.score, samples.target),
        _pearson(samples.score, samples.absolute_flow_change),
        top_fraction,
        top_precision,
        float(complex_better[score_mask].float().mean()),
        float(complex_better[~score_mask].float().mean()),
        float(samples.complex_advantage[score_mask].mean()),
        float(samples.complex_advantage[~score_mask].mean()),
    )


def save_routing_diagnostics(
    diagnostics: list[RoutingDiagnostic], path: str | Path
) -> None:
    if not diagnostics:
        raise ValueError("cannot save empty diagnostics")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(asdict(diagnostics[0])))
        writer.writeheader()
        writer.writerows(asdict(item) for item in diagnostics)


def save_routing_samples(samples: RoutingSamples, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(RoutingSamples.__dataclass_fields__)[1:]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=("date", *fields))
        writer.writeheader()
        for index, day in enumerate(samples.dates):
            writer.writerow(
                {"date": day, **{
                    field: float(getattr(samples, field)[index]) for field in fields
                }}
            )


def format_routing_diagnostics(diagnostics: list[RoutingDiagnostic]) -> str:
    lines = [
        "split       score~simple error  score~complex gain  score~physics gain  "
        "top-gain precision  complex-better high/low",
    ]
    for item in diagnostics:
        lines.append(
            f"{item.split:10s} {item.spearman_simple_error:+8.3f}            "
            f"{item.spearman_complex_advantage:+8.3f}            "
            f"{item.spearman_physics_advantage:+8.3f}             "
            f"{item.top_advantage_precision:8.3f}            "
            f"{item.complex_better_high_score:5.3f}/{item.complex_better_low_score:5.3f}"
        )
    lines.append("Correlations shown are Spearman; random top-gain precision equals top_fraction.")
    return "\n".join(lines)


def plot_routing_diagnostics(samples: RoutingSamples, path: str | Path) -> None:
    from .visualization import _finish_figure, _pyplot

    plt = _pyplot()
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    fields = (
        (samples.simple_error, "Simple squared error"),
        (samples.complex_advantage, "Complex error advantage"),
        (samples.physics_advantage, "Complex physics advantage"),
        (samples.target, "Observed discharge (mm/day)"),
        (samples.absolute_flow_change, "Absolute daily flow change"),
        (samples.weight, "Morse complex weight"),
    )
    score = samples.score.numpy()
    for axis, (values, title) in zip(axes.flat, fields):
        axis.scatter(score, values.numpy(), s=8, alpha=0.35)
        axis.set_xlabel("Morse score")
        axis.set_ylabel(title)
        axis.grid(alpha=0.2)
    figure.suptitle("Hydrology Morse-routing correlations")
    _finish_figure(figure, path)
