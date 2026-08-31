import argparse
from pathlib import Path

import torch

from complexity_ensemble.hydrology_data import load_camels_ch, load_ukraine_csv
from complexity_ensemble.hydrology_ude import make_hydrology_ude_data
from complexity_ensemble.hydrology_ude_comparison import (
    compare_hydrology_ude_models,
    hydrology_ude_summary,
    plot_hydrology_ude_comparison,
    save_hydrology_ude_comparison,
)


DEFAULT_CAMELS_CH_ROOT = Path("/mnt/c/Users/micro/Downloads/camels_ch/camels_ch")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mass-balanced hydrology universal differential equation")
    parser.add_argument("--source", choices=("camels_ch", "ukraine_csv"), default="camels_ch")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_CAMELS_CH_ROOT)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--basin", default="2011")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2020-12-31")
    parser.add_argument("--without-landcover", action="store_true")
    parser.add_argument("--simple", choices=("rbf", "fourier"), default="rbf")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--chunk-days", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hydrology/ude"))
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
    data = make_hydrology_ude_data(series)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    records = compare_hydrology_ude_models(
        data, simple_kind=args.simple,
        seeds=tuple(int(seed) for seed in args.seeds.split(",")),
        epochs=args.epochs, chunk_days=args.chunk_days,
        learning_rate=args.learning_rate, device=device,
    )
    stem = f"{args.source}_{args.basin}_{args.simple}_mlp_ude"
    save_hydrology_ude_comparison(records, args.output_dir / f"{stem}.csv")
    plot_hydrology_ude_comparison(records, args.output_dir / f"{stem}.png")
    print(hydrology_ude_summary(records))
    print(f"saved UDE results under {args.output_dir}")


if __name__ == "__main__":
    main()
