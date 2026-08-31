import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ensemble_review_matplotlib")

import matplotlib.pyplot as plt
import torch

from complexity_ensemble.pinnmamba import (
    PINNMamba,
    evaluate_reaction_pinnmamba,
    make_reaction_grid,
    reaction_losses,
)


def plot_evaluation(result, history: list[float], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    extent = [0.0, 2.0 * torch.pi, 1.0, 0.0]
    for axis, values, title in zip(
        axes.flat[:3],
        (result.reference, result.prediction, result.absolute_error),
        ("Exact reaction solution", "PINNMamba prediction", "Absolute error"),
    ):
        image = axis.imshow(values, extent=extent, aspect="auto", cmap="viridis")
        axis.set(xlabel="x", ylabel="t", title=title)
        figure.colorbar(image, ax=axis)
    axis = axes[1, 1]
    if history:
        axis.semilogy(history)
        axis.set(xlabel="L-BFGS outer step", ylabel="total loss", title="Training history")
    else:
        metrics = result.metrics
        labels = ["rMAE", "rRMSE"]
        ours = [metrics.relative_mae, metrics.relative_rmse]
        paper = [metrics.paper_relative_mae, metrics.paper_relative_rmse]
        positions = torch.arange(2).numpy()
        axis.bar(positions - 0.18, ours, width=0.36, label="evaluated")
        axis.bar(positions + 0.18, paper, width=0.36, label="paper")
        axis.set_xticks(positions, labels)
        axis.set_yscale("log")
        axis.set(title="Reaction benchmark comparison", ylabel="relative error")
        axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the ICML 2025 PINNMamba reaction benchmark")
    parser.add_argument("--checkpoint", type=Path, help="Official or locally trained state dict")
    parser.add_argument("--steps", type=int, default=0, help="L-BFGS outer steps; paper/repo uses 500")
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--boundary-weight", type=float, default=1.0,
                        help="1 matches released code; the paper text states 10")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pinnmamba"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PINNMamba().to(device)
    if args.checkpoint:
        model.load_official_checkpoint(args.checkpoint)
        model.to(device)

    history: list[float] = []
    if args.steps:
        grid = make_reaction_grid(args.grid_size, args.grid_size, device=device)
        optimizer = torch.optim.LBFGS(
            model.parameters(), line_search_fn="strong_wolfe",
            tolerance_grad=1e-8, tolerance_change=1e-10,
        )
        for step in range(1, args.steps + 1):
            latest = {}

            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                losses = reaction_losses(model, grid, boundary_weight=args.boundary_weight)
                losses.total.backward()
                latest["loss"] = float(losses.total.detach())
                return losses.total

            optimizer.step(closure)
            history.append(latest["loss"])
            if step == 1 or step % 10 == 0:
                print(f"step={step:4d} total={history[-1]:.3e}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.output_dir / "reaction_pinnmamba.pt")

    result = evaluate_reaction_pinnmamba(model)
    metrics = asdict(result.metrics)
    metrics["relative_mae_ratio_to_paper"] = result.metrics.relative_mae / result.metrics.paper_relative_mae
    metrics["relative_rmse_ratio_to_paper"] = result.metrics.relative_rmse / result.metrics.paper_relative_rmse
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reaction_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    plot_evaluation(result, history, args.output_dir / "reaction_comparison.png")
    print(json.dumps(metrics, indent=2))
    print(f"saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
