"""Selectable ensemble-training comparison for high-flow extreme events.

The public API in this module is deliberately independent of command-line and
data-loading concerns so it can also back an application UI later.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import torch

from .hydrology import HydrologyRoutedOutput, RoutedHydrologyModel
from .hydrology_consolidation_comparison import train_hydrology_model_with_consolidation
from .hydrology_data import HydrologyData, HydrologySplit
from .hydrology_distillation import train_hydrology_distill_to_simple
from .hydrology_hard_routing_consolidation_comparison import (
    train_hydrology_model_with_hard_routing_and_consolidation,
)


ENSEMBLE_APPROACHES = ("soft_routing", "hard_routing", "distillation")
SUPPORTED_ROUTERS = ("morse", "learned")


@dataclass(frozen=True)
class ExtremeComparisonConfig:
    """Model-independent and approach-specific settings for one comparison."""

    approaches: tuple[str, ...] = ENSEMBLE_APPROACHES
    routing: str = "morse"
    simple_kind: str = "rbf"
    complex_kind: str = "pinnmamba"
    seeds: tuple[int, ...] = (0, 1, 2)
    threshold_quantile: float = 0.95
    epochs: int = 20
    complex_epochs: int = 20
    distillation_epochs: int = 20
    consolidation_epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    training_noise: float = 0.0
    consolidation_weight: float = 0.05
    complexity_percentile: float = 80.0
    gate_temperature: float = 0.15
    hard_inference: bool = False
    include_baselines: bool = True

    def validate(self) -> None:
        unknown = set(self.approaches) - set(ENSEMBLE_APPROACHES)
        if not self.approaches or unknown:
            raise ValueError(
                "approaches must contain only: " + ", ".join(ENSEMBLE_APPROACHES)
            )
        if self.routing not in SUPPORTED_ROUTERS:
            raise ValueError("routing must be 'morse' or 'learned'")
        if "distillation" in self.approaches and self.routing == "learned":
            raise ValueError("distillation requires the fixed 'morse' router")
        if self.simple_kind not in {"rbf", "fourier"}:
            raise ValueError("simple_kind must be 'rbf' or 'fourier'")
        if self.complex_kind not in {"mlp", "pinnmamba"}:
            raise ValueError("complex_kind must be 'mlp' or 'pinnmamba'")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if not 0.0 < self.threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be strictly between zero and one")
        epoch_counts = (
            self.epochs,
            self.complex_epochs,
            self.distillation_epochs,
            self.consolidation_epochs,
        )
        if min(epoch_counts) < 1 or self.batch_size < 1 or self.learning_rate <= 0.0:
            raise ValueError("epoch counts, batch_size, and learning_rate must be positive")
        if self.training_noise < 0.0 or self.consolidation_weight < 0.0:
            raise ValueError("noise and consolidation weight must be non-negative")
        if not 0.0 < self.complexity_percentile < 100.0:
            raise ValueError("complexity_percentile must be strictly between zero and 100")
        if self.gate_temperature <= 0.0:
            raise ValueError("gate_temperature must be positive")


@dataclass(frozen=True)
class ExtremeEnsembleRecord:
    country: str
    basin: str
    approach: str
    routing: str
    inference_routing: str
    simple_expert: str
    complex_expert: str
    seed: int
    threshold_quantile: float
    threshold_mm_day: float
    test_days: int
    observed_extreme_days: int
    observed_events: int
    predicted_events: int
    precision: float
    recall: float
    critical_success_index: float
    false_alarm_ratio: float
    average_precision: float
    event_precision: float
    event_recall: float
    nse: float
    kge: float
    rmse_mm_day: float
    extreme_rmse_mm_day: float
    peak_magnitude_mae_mm_day: float
    peak_timing_mae_days: float
    mean_complex_weight: float
    parameters: int
    training_seconds: float
    simple_training_fraction: float


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _average_precision(scores: torch.Tensor, labels: torch.Tensor) -> float:
    scores = scores.detach().flatten().double().cpu()
    labels = labels.detach().flatten().bool().cpu()
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    group_end = torch.ones(len(scores), dtype=torch.bool)
    group_end[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    true_positive = sorted_labels.cumsum(0)[group_end].double()
    retrieved = torch.arange(1, len(scores) + 1, dtype=torch.double)[group_end]
    precision = true_positive / retrieved
    previous = torch.cat((torch.zeros(1, dtype=torch.double), true_positive[:-1]))
    return float((precision * (true_positive - previous) / positives).sum())


def _events(labels: torch.Tensor, dates: tuple[date, ...]) -> list[tuple[int, int]]:
    flags = labels.detach().flatten().bool().cpu().tolist()
    result: list[tuple[int, int]] = []
    start: int | None = None
    for index, positive in enumerate(flags):
        consecutive = (
            positive
            and start is not None
            and index > 0
            and dates[index] - dates[index - 1] == timedelta(days=1)
        )
        if positive and start is None:
            start = index
        elif positive and not consecutive:
            result.append((start, index - 1))
            start = index
        elif not positive and start is not None:
            result.append((start, index - 1))
            start = None
    if start is not None:
        result.append((start, len(flags) - 1))
    return result


def _overlaps(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return first[0] <= second[1] and second[0] <= first[1]


def _runoff_metrics(prediction: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    error = prediction - target
    denominator = (target - target.mean()).square().sum().clamp_min(1e-12)
    nse = 1.0 - error.square().sum() / denominator
    prediction_std = prediction.std(unbiased=False)
    target_std = target.std(unbiased=False)
    covariance = ((prediction - prediction.mean()) * (target - target.mean())).mean()
    correlation = covariance / (prediction_std * target_std).clamp_min(1e-12)
    alpha = prediction_std / target_std.clamp_min(1e-12)
    beta = prediction.mean() / target.mean().clamp_min(1e-12)
    kge = 1.0 - torch.sqrt(
        (correlation - 1.0).square() + (alpha - 1.0).square() + (beta - 1.0).square()
    )
    return float(nse), float(kge)


def extreme_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    dates: tuple[date, ...],
    threshold: float,
) -> dict[str, float | int]:
    """Return daily, event-level, and continuous runoff metrics."""
    prediction = prediction.detach().flatten().double().cpu()
    target = target.detach().flatten().double().cpu()
    observed = target > threshold
    predicted = prediction > threshold
    true_positive = int((observed & predicted).sum())
    false_positive = int((~observed & predicted).sum())
    false_negative = int((observed & ~predicted).sum())
    observed_events = _events(observed, dates)
    predicted_events = _events(predicted, dates)
    detected_observed = [
        item for item in observed_events
        if any(_overlaps(item, candidate) for candidate in predicted_events)
    ]
    detected_predicted = [
        item for item in predicted_events
        if any(_overlaps(item, candidate) for candidate in observed_events)
    ]
    peak_errors: list[float] = []
    timing_errors: list[float] = []
    for start, stop in detected_observed:
        actual = target[start : stop + 1]
        forecast = prediction[start : stop + 1]
        peak_errors.append(abs(float(forecast.max() - actual.max())))
        timing_errors.append(float(abs(int(forecast.argmax()) - int(actual.argmax()))))
    error = prediction - target
    extreme_error = error[observed]
    nse, kge = _runoff_metrics(prediction, target)
    return {
        "test_days": len(target),
        "observed_extreme_days": int(observed.sum()),
        "observed_events": len(observed_events),
        "predicted_events": len(predicted_events),
        "precision": _safe_ratio(true_positive, true_positive + false_positive),
        "recall": _safe_ratio(true_positive, true_positive + false_negative),
        "critical_success_index": _safe_ratio(
            true_positive, true_positive + false_positive + false_negative
        ),
        "false_alarm_ratio": _safe_ratio(false_positive, true_positive + false_positive),
        "average_precision": _average_precision(prediction, observed),
        "event_precision": _safe_ratio(len(detected_predicted), len(predicted_events)),
        "event_recall": _safe_ratio(len(detected_observed), len(observed_events)),
        "nse": nse,
        "kge": kge,
        "rmse_mm_day": float(error.square().mean().sqrt()),
        "extreme_rmse_mm_day": (
            float(extreme_error.square().mean().sqrt()) if len(extreme_error) else float("nan")
        ),
        "peak_magnitude_mae_mm_day": (
            sum(peak_errors) / len(peak_errors) if peak_errors else float("nan")
        ),
        "peak_timing_mae_days": (
            sum(timing_errors) / len(timing_errors) if timing_errors else float("nan")
        ),
    }


def _new_model(data: HydrologyData, config: ExtremeComparisonConfig) -> RoutedHydrologyModel:
    return RoutedHydrologyModel(
        data.feature_names,
        data.train.inputs.shape[1],
        data.discharge_scale,
        simple_kind=config.simple_kind,
        complex_kind=config.complex_kind,
        routing=config.routing,
        complexity_percentile=config.complexity_percentile,
        gate_temperature=config.gate_temperature,
    )


@torch.no_grad()
def _predict(
    model: RoutedHydrologyModel,
    split: HydrologySplit,
    discharge_scale: torch.Tensor,
    device: torch.device,
    *,
    hard: bool,
) -> tuple[torch.Tensor, float]:
    model.eval()
    output = model(split.inputs.to(device), hard=hard, return_details=True)
    assert isinstance(output, HydrologyRoutedOutput)
    prediction = output.discharge.squeeze(-1).cpu() * discharge_scale.cpu()
    return prediction, float(output.complex_weight.mean())


def _persistence_prediction(data: HydrologyData) -> torch.Tensor:
    flow_index = data.feature_names.index("previous_discharge")
    return data.test.physical_inputs[:, -1, flow_index].cpu()


def _climatology_prediction(data: HydrologyData) -> torch.Tensor:
    train_target = data.train.targets.cpu() * data.discharge_scale.cpu()
    grouped: dict[tuple[int, int], list[float]] = {}
    for day, value in zip(data.train.target_dates, train_target):
        grouped.setdefault((day.month, day.day), []).append(float(value))
    climatology = {key: sum(values) / len(values) for key, values in grouped.items()}
    fallback = float(train_target.mean())
    return torch.tensor(
        [climatology.get((day.month, day.day), fallback) for day in data.test.target_dates]
    )


def _record(
    data: HydrologyData,
    config: ExtremeComparisonConfig,
    *,
    approach: str,
    seed: int,
    threshold: float,
    prediction: torch.Tensor,
    inference_routing: str,
    mean_complex_weight: float,
    parameters: int,
    training_seconds: float,
    simple_training_fraction: float,
) -> ExtremeEnsembleRecord:
    target = data.test.targets.cpu() * data.discharge_scale.cpu()
    metrics = extreme_metrics(prediction, target, data.test.target_dates, threshold)
    return ExtremeEnsembleRecord(
        data.country,
        data.basin_id,
        approach,
        config.routing if approach not in {"persistence", "seasonal_climatology"} else "none",
        inference_routing,
        config.simple_kind if approach not in {"persistence", "seasonal_climatology"} else "none",
        config.complex_kind if approach not in {"persistence", "seasonal_climatology"} else "none",
        seed,
        config.threshold_quantile,
        threshold,
        **metrics,
        mean_complex_weight=mean_complex_weight,
        parameters=parameters,
        training_seconds=training_seconds,
        simple_training_fraction=simple_training_fraction,
    )


def compare_extreme_event_ensembles(
    data: HydrologyData,
    config: ExtremeComparisonConfig = ExtremeComparisonConfig(),
    *,
    device: torch.device | str = "cpu",
    verbose: bool = False,
) -> list[ExtremeEnsembleRecord]:
    """Train selected ensemble approaches and evaluate the same Q-threshold task."""
    config.validate()
    device = torch.device(device)
    train_target = data.train.targets.cpu() * data.discharge_scale.cpu()
    threshold = float(torch.quantile(train_target, config.threshold_quantile))
    records: list[ExtremeEnsembleRecord] = []
    if config.include_baselines:
        for approach, prediction in (
            ("persistence", _persistence_prediction(data)),
            ("seasonal_climatology", _climatology_prediction(data)),
        ):
            records.append(
                _record(
                    data,
                    config,
                    approach=approach,
                    seed=-1,
                    threshold=threshold,
                    prediction=prediction,
                    inference_routing="none",
                    mean_complex_weight=float("nan"),
                    parameters=0,
                    training_seconds=0.0,
                    simple_training_fraction=float("nan"),
                )
            )

    for seed in config.seeds:
        for approach in config.approaches:
            if verbose:
                print(f"training approach={approach} seed={seed}", flush=True)
            torch.manual_seed(seed)
            model = _new_model(data, config).to(device)
            started = time.perf_counter()
            simple_fraction = float("nan")
            if approach == "soft_routing":
                train_hydrology_model_with_consolidation(
                    model,
                    data.train,
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    training_noise=config.training_noise,
                    seed=seed,
                    device=device,
                    consolidation_weight=config.consolidation_weight,
                    feature_names=data.feature_names,
                    verbose=verbose,
                )
            elif approach == "hard_routing":
                train_hydrology_model_with_hard_routing_and_consolidation(
                    model,
                    data.train,
                    epochs=config.epochs,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    training_noise=config.training_noise,
                    seed=seed,
                    device=device,
                    consolidation_weight=config.consolidation_weight,
                    feature_names=data.feature_names,
                    verbose=verbose,
                )
            else:
                report = train_hydrology_distill_to_simple(
                    model,
                    data.train,
                    complex_epochs=config.complex_epochs,
                    distillation_epochs=config.distillation_epochs,
                    consolidation_epochs=config.consolidation_epochs,
                    batch_size=config.batch_size,
                    learning_rate=config.learning_rate,
                    training_noise=config.training_noise,
                    seed=seed,
                    device=device,
                    feature_names=data.feature_names,
                    consolidation_weight=config.consolidation_weight,
                    verbose=verbose,
                )
                simple_fraction = report.simple_fraction
            elapsed = time.perf_counter() - started
            hard_inference = approach == "hard_routing" and config.hard_inference
            prediction, mean_weight = _predict(
                model,
                data.test,
                data.discharge_scale,
                device,
                hard=hard_inference,
            )
            records.append(
                _record(
                    data,
                    config,
                    approach=approach,
                    seed=seed,
                    threshold=threshold,
                    prediction=prediction,
                    inference_routing="hard" if hard_inference else "soft",
                    mean_complex_weight=mean_weight,
                    parameters=sum(parameter.numel() for parameter in model.parameters()),
                    training_seconds=elapsed,
                    simple_training_fraction=simple_fraction,
                )
            )
    return records


def save_extreme_ensemble_records(
    records: list[ExtremeEnsembleRecord], path: str | Path
) -> None:
    if not records:
        raise ValueError("records must not be empty")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for record in records]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_extreme_ensemble_summary(records: list[ExtremeEnsembleRecord]) -> str:
    header = "approach              seed     AP  recall    CSI  event-R  extreme-RMSE"
    lines = [header]
    for record in records:
        values = (
            record.average_precision,
            record.recall,
            record.critical_success_index,
            record.event_recall,
            record.extreme_rmse_mm_day,
        )
        rendered = ["nan" if not math.isfinite(value) else f"{value:.3f}" for value in values]
        lines.append(
            f"{record.approach:<21} {record.seed:>4}  {rendered[0]:>5}  "
            f"{rendered[1]:>6}  {rendered[2]:>5}  {rendered[3]:>7}  {rendered[4]:>12}"
        )
    return "\n".join(lines)


__all__ = [
    "ENSEMBLE_APPROACHES",
    "ExtremeComparisonConfig",
    "ExtremeEnsembleRecord",
    "compare_extreme_event_ensembles",
    "extreme_metrics",
    "format_extreme_ensemble_summary",
    "save_extreme_ensemble_records",
]
