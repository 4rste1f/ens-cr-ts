import argparse
from pathlib import Path

import torch

from complexity_ensemble.hydrology import RoutedHydrologyModel
from complexity_ensemble.hydrology_comparison import train_hydrology_model
from complexity_ensemble.hydrology_data import load_camels_ch, load_ukraine_csv, make_hydrology_data
from complexity_ensemble.hydrology_diagnostics import (
    collect_routing_samples,
    format_routing_diagnostics,
    plot_routing_diagnostics,
    routing_diagnostic,
    save_routing_diagnostics,
    save_routing_samples,
)


DEFAULT_CAMELS_CH_ROOT = Path("/mnt/c/Users/micro/Downloads/camels_ch/camels_ch")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test whether hydrologic Morse scores route useful cases")
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-fraction", type=float, default=0.2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hydrology/diagnostics"))
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
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    torch.manual_seed(args.seed)
    model = RoutedHydrologyModel(
        data.feature_names, args.sequence_length, data.discharge_scale,
        simple_kind=args.simple, complex_kind=args.complex, routing="morse",
    ).to(device)
    train_hydrology_model(
        model, data.train, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, training_noise=0.0,
        seed=args.seed, device=device,
    )

    diagnostics = []
    stem = f"{args.source}_{args.basin}_{args.simple}_{args.complex}_routing"
    for split_name, split in (("validation", data.validation), ("test", data.test)):
        samples = collect_routing_samples(model, split, data, device=device)
        diagnostics.append(
            routing_diagnostic(samples, split=split_name, top_fraction=args.top_fraction)
        )
        save_routing_samples(samples, args.output_dir / f"{stem}_{split_name}_samples.csv")
        if split_name == "test":
            plot_routing_diagnostics(samples, args.output_dir / f"{stem}.png")
    save_routing_diagnostics(diagnostics, args.output_dir / f"{stem}_summary.csv")
    print(format_routing_diagnostics(diagnostics))
    print(f"saved diagnostics under {args.output_dir}")


if __name__ == "__main__":
    main()
