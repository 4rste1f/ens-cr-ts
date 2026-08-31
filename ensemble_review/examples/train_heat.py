import argparse
from pathlib import Path

import torch

from complexity_ensemble.evaluation import evaluate_heat_pinn, format_metrics, save_metrics
from complexity_ensemble.pinn import HeatPINN, sample_heat_problem
from complexity_ensemble.visualization import plot_pinn_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simple", choices=("rbf", "fourier"), default="rbf")
    parser.add_argument("--complex", choices=("mlp", "pinnmamba"), default="mlp")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = sample_heat_problem(device=device)
    model = HeatPINN(simple_kind=args.simple, complex_kind=args.complex).to(device)
    model.fit_router(torch.cat((data[0], data[2], data[4]), dim=0))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history: dict[str, list[float]] = {
        "total": [], "physics": [], "interface": [], "alignment": []
    }

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        losses = model.losses(*data, interface_weight=0.05)
        losses.total.backward()
        optimizer.step()
        history["total"].append(losses.total.item())
        history["physics"].append(losses.physics.item())
        history["interface"].append(losses.interface.item())
        history["alignment"].append(losses.alignment.item())
        if epoch == 1 or epoch % 100 == 0:
            print(
                f"epoch={epoch:4d} total={losses.total.item():.3e} "
                f"physics={losses.physics.item():.3e} complex={losses.complex_usage.item():.2%}"
            )

    result = evaluate_heat_pinn(model)
    run_name = f"heat_{args.simple}_{args.complex}"
    save_metrics(result.metrics, args.output_dir / f"{run_name}_metrics.json")
    plot_pinn_evaluation(result, history, args.output_dir / f"{run_name}.png")
    print("\nQuality metrics\n" + format_metrics(result.metrics))
    print(f"saved {args.output_dir / f'{run_name}.png'}")


if __name__ == "__main__":
    main()
