import argparse
from pathlib import Path

import torch

from complexity_ensemble.hydrology_comparison import (
    compare_hydrology_models,
    hydrology_comparison_summary,
    plot_hydrology_comparison,
    save_hydrology_comparison,
)
from complexity_ensemble.hydrology_data import load_camels_ch, load_ukraine_csv, make_hydrology_data
from complexity_ensemble.hydrology_morse_learning import (
    assess_learned_morse,
    calibrate_learned_morse,
    save_calibration_report,
    save_potential_assessment,
)


DEFAULT_CAMELS_CH_ROOT = Path("/mnt/c/Users/micro/Downloads/camels_ch/camels_ch")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare analytic and data-calibrated Morse routing")
    parser.add_argument("--source", choices=("camels_ch", "ukraine_csv"), default="camels_ch")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_CAMELS_CH_ROOT)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--basin", default="2011")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2020-12-31")
    parser.add_argument("--without-landcover", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--simple", choices=("rbf", "fourier"), default="rbf")
    parser.add_argument("--complex", choices=("mlp", "pinnmamba"), default="mlp")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--pilot-epochs", type=int, default=5)
    parser.add_argument("--potential-epochs", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20, help="Final ensemble epochs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--training-noise", type=float, default=0.1,
        help="Noise standard deviation for dynamic training inputs (default: 0.1)",
    )
    parser.add_argument(
        "--inference-noise", type=float, default=0.1,
        help="Noise standard deviation for dynamic inference inputs (default: 0.1)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hydrology/learned_morse"))
    args = parser.parse_args()
    if args.training_noise < 0.0 or args.inference_noise < 0.0:
        parser.error("noise levels must be non-negative")

    if args.source == "camels_ch":
        series = load_camels_ch(
            args.data_root, args.basin, start=args.start, end=args.end,
            include_landcover=not args.without_landcover,
        )
    else:
        if args.csv is None:
            parser.error("--csv is required with --source ukraine_csv")
        series = load_ukraine_csv(args.csv, basin_id=args.basin)
    data = make_hydrology_data(series, sequence_length=args.sequence_length)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    potential, calibration = calibrate_learned_morse(
        data, simple_kind=args.simple, complex_kind=args.complex,
        folds=args.folds, pilot_epochs=args.pilot_epochs,
        potential_epochs=args.potential_epochs, batch_size=args.batch_size,
        seed=args.seed, device=device,
    )
    assessment = assess_learned_morse(
        data, potential, simple_kind=args.simple, complex_kind=args.complex,
        pilot_epochs=args.pilot_epochs, batch_size=args.batch_size,
        seed=args.seed, device=device,
    )
    records = compare_hydrology_models(
        data, simple_kind=args.simple, complex_kind=args.complex,
        seeds=(args.seed,), training_noise_levels=(0.0, args.training_noise),
        inference_noise_levels=(0.0, args.inference_noise), epochs=args.epochs,
        batch_size=args.batch_size, device=device,
        learned_morse_potential=potential,
    )
    stem = f"{args.source}_{args.basin}_{args.simple}_{args.complex}_learned_morse"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_calibration_report(calibration, args.output_dir / f"{stem}_calibration.csv")
    save_potential_assessment(assessment, args.output_dir / f"{stem}_transfer.csv")
    save_hydrology_comparison(records, args.output_dir / f"{stem}_comparison.csv")
    plot_hydrology_comparison(records, args.output_dir / f"{stem}.png")
    torch.save(potential.state_dict(), args.output_dir / f"{stem}_potential.pt")
    print(
        f"calibration samples={calibration.samples}, "
        f"positive utility={calibration.positive_utility_fraction:.3f}, "
        f"score/utility Spearman={calibration.score_utility_spearman:+.3f}"
    )
    for item in assessment:
        print(
            f"{item.split} transfer: score/utility Spearman="
            f"{item.score_utility_spearman:+.3f}, top-20% precision="
            f"{item.top_utility_precision:.3f}, complex-useful high/low="
            f"{item.positive_utility_high_score:.3f}/{item.positive_utility_low_score:.3f}"
        )
    print(hydrology_comparison_summary(records))
    print(f"saved learned-Morse results under {args.output_dir}")


if __name__ == "__main__":
    main()
