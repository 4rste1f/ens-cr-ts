import argparse
from pathlib import Path

import torch

from complexity_ensemble.comparison import (
    compare_nde_models,
    comparison_summary,
    plot_comparison,
    save_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simple", choices=("rbf", "fourier"), default="rbf")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--training-noise", "--noise", dest="noise", type=float, default=0.05)
    parser.add_argument("--inference-noise", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/comparisons"))
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = compare_nde_models(
        args.simple,
        seeds=seeds,
        noise_levels=(0.0, args.noise),
        inference_noise_levels=(0.0, args.inference_noise),
        epochs=args.epochs,
        device=device,
    )
    stem = f"pendulum_{args.simple}_comparison"
    save_comparison(records, args.output_dir / f"{stem}.csv")
    plot_comparison(records, args.output_dir / f"{stem}.png")
    print(comparison_summary(records))
    print(f"saved results under {args.output_dir}")


if __name__ == "__main__":
    main()
