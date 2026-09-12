"""Publication-oriented hydrology robustness experiment.

Unlike the original consolidation example, this entry point applies one noise
realization per dated observation.  Consequently, overlapping windows and the
expert/router views of a window contain the same corrupted measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, stdev
from typing import Mapping

import torch

from complexity_ensemble.hydrology import LearnedHydrologyMorsePotential
from complexity_ensemble.hydrology_comparison import (
    _complexity_diagnostics,
    _estimate_complexity_features,
    _evaluate,
    _make_model,
    _parameter_count,
    _prepare_complexity_estimators,
    _select_models,
    hydrology_comparison_summary,
)
from complexity_ensemble.hydrology_complexity import (
    MODEL_NAMES,
    HydrologyComplexityEstimator,
    LyapunovComplexityEstimator,
    TakensPersistenceEstimator,
)
from complexity_ensemble.hydrology_consolidation_comparison import (
    train_hydrology_model_with_consolidation,
)
from complexity_ensemble.hydrology_data import (
    HydrologyData,
    HydrologySeries,
    HydrologySplit,
    load_camels_ch,
    load_ukraine_csv,
    make_hydrology_data,
)
from complexity_ensemble.hydrology_morse_learning import (
    calibrate_learned_morse,
    save_calibration_report,
)


DEFAULT_CAMELS_CH_ROOT = Path("/mnt/c/Users/micro/Downloads/camels_ch/camels_ch")
DYNAMIC_NOISE_FEATURES = {"precipitation", "temperature", "pet", "previous_discharge"}
NONNEGATIVE_NOISE_FEATURES = {"precipitation", "pet", "previous_discharge"}
DEFAULT_MODELS = ("morse", "learned_morse", "learned", "single_complex")


@dataclass(frozen=True)
class EvaluationPeriod:
    label: str
    start: str
    end: str


@dataclass(frozen=True)
class RobustnessRecord:
    country: str
    basin: str
    period: str
    period_start: str
    period_end: str
    model: str
    simple_expert: str
    complex_expert: str
    seed: int
    training_noise_seed: int
    inference_noise_seed: int
    training_noise: float
    inference_noise: float
    validation_nse: float
    test_nse: float
    test_kge: float
    test_rmse_mm_day: float
    physics_error: float
    parameters: int
    mean_complex_weight: float
    mean_complexity_score: float
    valid_complexity_fraction: float
    consolidation_weight: float


def parse_csv_values(value: str, cast: type, name: str) -> tuple:
    try:
        values = tuple(dict.fromkeys(cast(item.strip()) for item in value.split(",") if item.strip()))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid {name}: {error}") from error
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one value")
    return values


def parse_period(value: str) -> EvaluationPeriod:
    parts = value.split(":")
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("period must have the form LABEL:YYYY-MM-DD:YYYY-MM-DD")
    label, start, end = parts
    try:
        start_date, end_date = date.fromisoformat(start), date.fromisoformat(end)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid period date: {error}") from error
    if end_date <= start_date:
        raise argparse.ArgumentTypeError("period end must be later than period start")
    return EvaluationPeriod(label, start, end)


def _window_dates(target_date: date, length: int) -> tuple[date, ...]:
    return tuple(target_date - timedelta(days=lag) for lag in range(length, 0, -1))


def coherently_corrupt_split(
    split: HydrologySplit,
    data: HydrologyData,
    noise: float,
    *,
    seed: int,
) -> HydrologySplit:
    """Apply standardized Gaussian noise once per date and dynamic feature.

    The clean targets and physical-loss inputs are deliberately retained.  This
    isolates imperfect model observations while keeping the verification target
    and physics constraint fixed.  Nonnegative variables are clipped at zero in
    physical units after corruption.
    """
    if noise < 0.0:
        raise ValueError("noise must be non-negative")
    if noise == 0.0:
        return split

    input_dates = {
        day
        for target_date in split.target_dates
        for day in _window_dates(target_date, split.inputs.shape[1])
    }
    if split.routing_inputs is not None:
        input_dates.update(
            day
            for target_date in split.target_dates
            for day in _window_dates(target_date, split.routing_inputs.shape[1])
        )
    ordered_dates = sorted(input_dates)
    generator = torch.Generator().manual_seed(seed)
    innovations = torch.randn(
        (len(ordered_dates), len(data.feature_names)), generator=generator,
        dtype=split.inputs.dtype,
    )
    dynamic_mask = torch.tensor(
        [name in DYNAMIC_NOISE_FEATURES for name in data.feature_names],
        dtype=split.inputs.dtype,
    )
    innovations *= dynamic_mask
    noise_by_date = {day: innovations[index] for index, day in enumerate(ordered_dates)}

    lower_bounds = torch.full_like(data.feature_mean, -torch.inf)
    for index, feature_name in enumerate(data.feature_names):
        if feature_name in NONNEGATIVE_NOISE_FEATURES:
            lower_bounds[index] = -data.feature_mean[index] / data.feature_scale[index]

    def corrupt(windows: torch.Tensor) -> torch.Tensor:
        result = windows.clone()
        for row, target_date in enumerate(split.target_dates):
            dates = _window_dates(target_date, windows.shape[1])
            perturbation = torch.stack([noise_by_date[day] for day in dates])
            result[row] = torch.maximum(result[row] + noise * perturbation, lower_bounds)
        return result

    return replace(
        split,
        inputs=corrupt(split.inputs),
        routing_inputs=None if split.routing_inputs is None else corrupt(split.routing_inputs),
    )


def compare_hydrology_models_dose_response(
    data: HydrologyData,
    *,
    period: EvaluationPeriod,
    models: tuple[str, ...],
    simple_kind: str,
    complex_kind: str,
    seeds: tuple[int, ...],
    training_noise_levels: tuple[float, ...],
    inference_noise_levels: tuple[float, ...],
    inference_noise_draws: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    consolidation_weight: float,
    device: torch.device,
    learned_morse_potential: LearnedHydrologyMorsePotential | None,
    complexity_estimators: Mapping[str, HydrologyComplexityEstimator] | None = None,
    complexity_percentile: float = 80.0,
    gate_temperature: float = 0.15,
) -> list[RobustnessRecord]:
    """Train and evaluate a paired, coherent noise dose-response grid."""
    if inference_noise_draws < 1:
        raise ValueError("inference_noise_draws must be positive")
    if any(level < 0.0 for level in (*training_noise_levels, *inference_noise_levels)):
        raise ValueError("noise levels must be non-negative")
    selected_models = _select_models(models, learned_morse_potential)
    records: list[RobustnessRecord] = []

    for seed in seeds:
        training_noise_seed = seed + 100003
        for training_noise in sorted(set(training_noise_levels)):
            noisy_train = coherently_corrupt_split(
                data.train, data, training_noise, seed=training_noise_seed
            )
            training_data = replace(data, train=noisy_train)
            prepared_estimators = _prepare_complexity_estimators(
                training_data, selected_models, complexity_estimators
            )
            train_features = {
                model_name: _estimate_complexity_features(
                    prepared_estimators.get(model_name), noisy_train, data.feature_names,
                    noise=0.0, seed=training_noise_seed,
                )
                for model_name in selected_models
            }
            for model_name in selected_models:
                torch.manual_seed(seed)
                model = _make_model(
                    model_name, training_data, simple_kind, complex_kind,
                    learned_morse_potential, complexity_percentile, gate_temperature,
                ).to(device)
                train_hydrology_model_with_consolidation(
                    model, noisy_train, epochs=epochs, batch_size=batch_size,
                    learning_rate=learning_rate, training_noise=0.0, seed=seed,
                    device=device, consolidation_weight=consolidation_weight,
                    feature_names=data.feature_names,
                    routing_features=train_features[model_name],
                )

                for draw in range(inference_noise_draws):
                    inference_noise_seed = seed + 300007 + 1000003 * draw
                    for inference_noise in sorted(set(inference_noise_levels)):
                        noisy_validation = coherently_corrupt_split(
                            data.validation, data, inference_noise,
                            seed=inference_noise_seed + 17,
                        )
                        noisy_test = coherently_corrupt_split(
                            data.test, data, inference_noise, seed=inference_noise_seed
                        )
                        validation_features = _estimate_complexity_features(
                            prepared_estimators.get(model_name), noisy_validation,
                            data.feature_names, noise=0.0, seed=inference_noise_seed + 17,
                        )
                        test_features = _estimate_complexity_features(
                            prepared_estimators.get(model_name), noisy_test,
                            data.feature_names, noise=0.0, seed=inference_noise_seed,
                        )
                        validation_nse, _, _, _, _ = _evaluate(
                            model, noisy_validation, data, inference_noise=0.0,
                            seed=inference_noise_seed + 17, device=device,
                            routing_features=validation_features,
                        )
                        test_nse, test_kge, test_rmse, physics, usage = _evaluate(
                            model, noisy_test, data, inference_noise=0.0,
                            seed=inference_noise_seed, device=device,
                            routing_features=test_features,
                        )
                        mean_score, valid_fraction = _complexity_diagnostics(test_features)
                        records.append(
                            RobustnessRecord(
                                data.country, data.basin_id, period.label, period.start,
                                period.end, model_name,
                                simple_kind if model_name != "single_complex" else "none",
                                complex_kind, seed, training_noise_seed,
                                inference_noise_seed, training_noise, inference_noise,
                                validation_nse, test_nse, test_kge, test_rmse, physics,
                                _parameter_count(model), usage, mean_score, valid_fraction,
                                consolidation_weight,
                            )
                        )
    return records


def save_records(records: list[RobustnessRecord], path: Path) -> None:
    if not records:
        raise ValueError("cannot save empty robustness results")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def plot_dose_response(records: list[RobustnessRecord], path: Path) -> None:
    """Plot inference-noise response curves separately for each training dose."""
    if not records:
        raise ValueError("cannot plot empty robustness results")
    from complexity_ensemble.visualization import _finish_figure, _pyplot

    models = tuple(
        model for model in MODEL_NAMES if any(record.model == model for record in records)
    )
    training_levels = sorted({record.training_noise for record in records})
    inference_levels = sorted({record.inference_noise for record in records})
    metrics = (
        ("test_nse", "Test NSE"),
        ("test_kge", "Test KGE"),
        ("test_rmse_mm_day", "Test RMSE (mm/day)"),
    )
    plt = _pyplot()
    figure, axes = plt.subplots(
        len(metrics), len(training_levels),
        figsize=(4.2 * len(training_levels), 3.4 * len(metrics)),
        squeeze=False, sharex=True,
    )
    for column, training_noise in enumerate(training_levels):
        for row, (field, label) in enumerate(metrics):
            axis = axes[row][column]
            for model in models:
                groups = [
                    [
                        getattr(record, field)
                        for record in records
                        if record.model == model
                        and record.training_noise == training_noise
                        and record.inference_noise == inference_noise
                    ]
                    for inference_noise in inference_levels
                ]
                averages = [mean(group) for group in groups]
                spreads = [stdev(group) if len(group) > 1 else 0.0 for group in groups]
                line = axis.plot(inference_levels, averages, marker="o", label=model)[0]
                axis.fill_between(
                    inference_levels,
                    [average - spread for average, spread in zip(averages, spreads)],
                    [average + spread for average, spread in zip(averages, spreads)],
                    color=line.get_color(), alpha=0.12,
                )
            if row == 0:
                axis.set_title(f"Training noise $\\sigma={training_noise:g}$")
            if column == 0:
                axis.set_ylabel(label)
            if row == len(metrics) - 1:
                axis.set_xlabel("Inference noise $\\sigma$ (training-standardized units)")
            axis.grid(alpha=0.25)
    axes[0][-1].legend(fontsize=8)
    first = records[0]
    figure.suptitle(
        f"{first.country} basin {first.basin}, period {first.period}: noise dose response"
    )
    _finish_figure(figure, path)


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, EvaluationPeriod):
        return asdict(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _complexity_estimators(args: argparse.Namespace, data: HydrologyData) -> dict:
    flow_index = data.feature_names.index("previous_discharge")
    result = {}
    if "tda" in args.models:
        result["tda"] = TakensPersistenceEstimator(
            flow_index, embedding_dim=args.tda_embedding_dim, delay=args.tda_delay,
            min_persistence=args.tda_min_persistence, score=args.tda_score,
            context_length=args.tda_context_length,
        )
    if "lyapunov" in args.models:
        result["lyapunov"] = LyapunovComplexityEstimator(
            flow_index, embedding_dim=args.lyapunov_embedding_dim,
            delay=args.lyapunov_delay, theiler_window=args.lyapunov_theiler_window,
            fit_horizon=args.lyapunov_fit_horizon, min_r2=args.lyapunov_min_r2,
            context_length=args.lyapunov_context_length,
        )
    return result


def _load_series(args: argparse.Namespace, basin: str, period: EvaluationPeriod) -> HydrologySeries:
    if args.source == "camels_ch":
        return load_camels_ch(
            args.data_root, basin, start=period.start, end=period.end,
            include_landcover=not args.without_landcover,
        )
    if args.csv_template is not None:
        path = Path(str(args.csv_template).format(basin=basin))
    elif args.csv is not None and len(args.basins) == 1:
        path = args.csv
    else:
        raise ValueError(
            "Ukraine multi-basin runs require --csv-template containing {basin}; "
            "--csv is supported for a single basin"
        )
    series = load_ukraine_csv(path, basin_id=basin)
    start, end = date.fromisoformat(period.start), date.fromisoformat(period.end)
    keep = [index for index, day in enumerate(series.dates) if start <= day <= end]
    if not keep:
        raise ValueError(f"basin {basin} has no records in period {period.label}")
    indices = torch.tensor(keep)
    return replace(
        series, dates=tuple(series.dates[index] for index in keep),
        discharge=series.discharge[indices], forcings=series.forcings[indices],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-basin, multi-period coherent-noise hydrology robustness study"
    )
    parser.add_argument("--source", choices=("camels_ch", "ukraine_csv"), default="camels_ch")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_CAMELS_CH_ROOT)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--csv-template", type=Path)
    parser.add_argument("--basins", default="2011")
    parser.add_argument(
        "--period", action="append", type=parse_period,
        help="Repeatable LABEL:START:END period (default: full:2000-01-01:2020-12-31)",
    )
    parser.add_argument("--without-landcover", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--tda-context-length", type=int, default=128)
    parser.add_argument("--tda-embedding-dim", type=int, default=3)
    parser.add_argument("--tda-delay", type=int, default=7)
    parser.add_argument("--tda-min-persistence", type=float, default=0.1)
    parser.add_argument(
        "--tda-score", choices=("total_persistence", "persistent_holes", "persistence_entropy"),
        default="total_persistence",
    )
    parser.add_argument("--lyapunov-context-length", type=int, default=365)
    parser.add_argument("--lyapunov-embedding-dim", type=int, default=3)
    parser.add_argument("--lyapunov-delay", type=int, default=7)
    parser.add_argument("--lyapunov-theiler-window", type=int, default=30)
    parser.add_argument("--lyapunov-fit-horizon", type=int, default=20)
    parser.add_argument("--lyapunov-min-r2", type=float, default=0.8)
    parser.add_argument("--complexity-percentile", type=float, default=80.0)
    parser.add_argument("--gate-temperature", type=float, default=0.15)
    parser.add_argument("--simple", choices=("rbf", "fourier"), default="rbf")
    parser.add_argument("--complex", choices=("mlp", "pinnmamba"), default="mlp")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--pilot-epochs", type=int, default=5)
    parser.add_argument("--potential-epochs", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--training-noise-levels", default="0,0.05,0.1,0.2")
    parser.add_argument("--inference-noise-levels", default="0,0.025,0.05,0.1,0.2,0.4")
    parser.add_argument("--inference-noise-draws", type=int, default=10)
    parser.add_argument("--consolidation-weight", type=float, default=0.05)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/hydrology/coherent_noise_robustness"),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.basins = parse_csv_values(args.basins, str, "--basins")
    args.models = parse_csv_values(args.models, str, "--models")
    args.seeds = parse_csv_values(args.seeds, int, "--seeds")
    args.training_noise_levels = parse_csv_values(
        args.training_noise_levels, float, "--training-noise-levels"
    )
    args.inference_noise_levels = parse_csv_values(
        args.inference_noise_levels, float, "--inference-noise-levels"
    )
    args.period = tuple(args.period or [EvaluationPeriod("full", "2000-01-01", "2020-12-31")])
    invalid_models = set(args.models) - set(MODEL_NAMES)
    if invalid_models:
        parser.error(f"unknown models: {', '.join(sorted(invalid_models))}")
    if any(level < 0 for level in (*args.training_noise_levels, *args.inference_noise_levels)):
        parser.error("noise levels must be non-negative")
    if args.epochs < 1 or args.inference_noise_draws < 1 or args.consolidation_weight < 0:
        parser.error("epochs/draws must be positive and consolidation weight non-negative")

    routing_length = max(
        [args.sequence_length]
        + ([args.tda_context_length] if "tda" in args.models else [])
        + ([args.lyapunov_context_length] if "lyapunov" in args.models else [])
    )
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {key: _jsonable(value) for key, value in vars(args).items()}
    with (args.output_dir / "configuration.json").open("w", encoding="utf-8") as output:
        json.dump(configuration, output, indent=2, sort_keys=True)
        output.write("\n")

    all_records: list[RobustnessRecord] = []
    failures = []
    for basin in args.basins:
        for period in args.period:
            run_name = f"{args.source}_{basin}_{period.label}"
            try:
                series = _load_series(args, basin, period)
                data = make_hydrology_data(
                    series, sequence_length=args.sequence_length,
                    routing_context_length=routing_length,
                )
                potential = None
                if "learned_morse" in args.models:
                    potential, calibration = calibrate_learned_morse(
                        data, simple_kind=args.simple, complex_kind=args.complex,
                        folds=args.folds, pilot_epochs=args.pilot_epochs,
                        potential_epochs=args.potential_epochs, batch_size=args.batch_size,
                        seed=args.seeds[0], device=device,
                    )
                    save_calibration_report(
                        calibration, args.output_dir / f"{run_name}_calibration.csv"
                    )
                    torch.save(potential.state_dict(), args.output_dir / f"{run_name}_potential.pt")
                records = compare_hydrology_models_dose_response(
                    data, period=period, models=args.models, simple_kind=args.simple,
                    complex_kind=args.complex, seeds=args.seeds,
                    training_noise_levels=args.training_noise_levels,
                    inference_noise_levels=args.inference_noise_levels,
                    inference_noise_draws=args.inference_noise_draws,
                    epochs=args.epochs, batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    consolidation_weight=args.consolidation_weight, device=device,
                    learned_morse_potential=potential,
                    complexity_estimators=_complexity_estimators(args, data),
                    complexity_percentile=args.complexity_percentile,
                    gate_temperature=args.gate_temperature,
                )
                all_records.extend(records)
                save_records(records, args.output_dir / f"{run_name}_comparison.csv")
                plot_dose_response(records, args.output_dir / f"{run_name}.png")
                print(hydrology_comparison_summary(records))
            except Exception as error:
                if not args.continue_on_error:
                    raise
                failures.append({"basin": basin, "period": period.label, "error": repr(error)})
                print(f"FAILED {run_name}: {error}")

    if all_records:
        save_records(all_records, args.output_dir / "all_comparisons.csv")
    if failures:
        with (args.output_dir / "failures.json").open("w", encoding="utf-8") as output:
            json.dump(failures, output, indent=2)
            output.write("\n")
    if not all_records:
        raise RuntimeError("no basin-period run completed")
    print(f"saved {len(all_records)} records under {args.output_dir}")


if __name__ == "__main__":
    main()
