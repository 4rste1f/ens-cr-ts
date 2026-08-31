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


DEFAULT_CAMELS_CH_ROOT = Path("/mnt/c/Users/micro/Downloads/camels_ch/camels_ch")


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-compute rainfall-runoff comparison")
    parser.add_argument("--source", choices=("camels_ch", "ukraine_csv"), default="camels_ch")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_CAMELS_CH_ROOT)
    parser.add_argument("--csv", type=Path, help="Canonical Ukrainian daily CSV")
    parser.add_argument("--basin", default="2011")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2020-12-31")
    parser.add_argument("--without-landcover", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--simple", choices=("rbf", "fourier"), default="rbf")
    parser.add_argument("--complex", choices=("mlp", "pinnmamba"), default="mlp")
    parser.add_argument(
        "--models", default="morse,learned,single_complex",
        help="Comma-separated models to train: morse, learned, single_complex",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seeds", default="123")
    parser.add_argument("--training-noise", type=float, default=0.0)
    parser.add_argument("--inference-noise", type=float, default=0.1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hydrology"))
    args = parser.parse_args()

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
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    records = compare_hydrology_models(
        data, models=tuple(model.strip() for model in args.models.split(",") if model.strip()),
        simple_kind=args.simple, complex_kind=args.complex,
        seeds=tuple(int(seed) for seed in args.seeds.split(",")),
        training_noise_levels=(0.0, args.training_noise),
        inference_noise_levels=(0.0, args.inference_noise),
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, device=device,
    )
    stem = f"{args.source}_{args.basin}_{args.simple}_{args.complex}_comparison"
    save_hydrology_comparison(records, args.output_dir / f"{stem}.csv")
    plot_hydrology_comparison(records, args.output_dir / f"{stem}.png")
    print(hydrology_comparison_summary(records))
    print(f"saved results under {args.output_dir}")


if __name__ == "__main__":
    main()
