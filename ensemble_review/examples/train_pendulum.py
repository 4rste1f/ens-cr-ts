import argparse
from pathlib import Path

import torch

from complexity_ensemble.nde import MorseRoutedVectorField, sample_pendulum_derivatives
from complexity_ensemble.evaluation import evaluate_pendulum_nde, format_metrics, save_metrics
from complexity_ensemble.visualization import plot_nde_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simple", choices=("rbf", "fourier"), default="rbf")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs, target = sample_pendulum_derivatives(device=device)
    model = MorseRoutedVectorField(state_dim=2, simple_kind=args.simple).to(device)
    model.router.fit(inputs)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history: dict[str, list[float]] = {
        "total": [],
        "data": [],
        "kinematic": [],
        "interface": [],
    }

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        losses = model.losses(inputs, target, interface_weight=0.05)
        losses.total.backward()
        optimizer.step()
        history["total"].append(losses.total.item())
        history["data"].append(losses.data.item())
        history["kinematic"].append(losses.kinematic_physics.item())
        history["interface"].append(losses.interface.item())
        if epoch == 1 or epoch % 100 == 0:
            print(
                f"epoch={epoch:4d} total={losses.total.item():.3e} "
                f"kinematic={losses.kinematic_physics.item():.3e} "
                f"complex={losses.complex_usage.item():.2%}"
            )

    result = evaluate_pendulum_nde(model)
    run_name = f"pendulum_{args.simple}"
    save_metrics(result.metrics, args.output_dir / f"{run_name}_metrics.json")
    plot_nde_evaluation(result, history, args.output_dir / f"{run_name}.png")
    print("\nQuality metrics\n" + format_metrics(result.metrics))
    print(f"saved {args.output_dir / f'{run_name}.png'}")


if __name__ == "__main__":
    main()
