from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import tempfile

from .evaluation import NDEEvaluation, PINNEvaluation


def _pyplot():
    cache = Path(tempfile.gettempdir()) / "complexity-ensemble-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt

    return plt


def _finish_figure(figure: object, path: str | Path) -> None:
    plt = _pyplot()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_pinn_evaluation(
    result: PINNEvaluation,
    history: Mapping[str, Sequence[float]],
    path: str | Path,
) -> None:
    plt = _pyplot()

    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    fields = (
        (result.prediction, "Routed prediction", "viridis"),
        (result.reference, "Analytical solution", "viridis"),
        (result.absolute_error, "Absolute error", "magma"),
        (result.complex_weight, "Complex-expert weight", "coolwarm"),
        (result.complexity, "Morse complexity", "cividis"),
    )
    for axis, (field, title, cmap) in zip(axes.flat[:5], fields):
        image = axis.pcolormesh(result.t_grid, result.x_grid, field, shading="auto", cmap=cmap)
        axis.set_title(title)
        axis.set_xlabel("time")
        axis.set_ylabel("space")
        figure.colorbar(image, ax=axis)

    loss_axis = axes.flat[5]
    for name, values in history.items():
        if values:
            loss_axis.semilogy(range(1, len(values) + 1), values, label=name)
    loss_axis.set_title("Training losses")
    loss_axis.set_xlabel("epoch")
    loss_axis.set_ylabel("loss")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend()
    figure.suptitle("Morse-routed heat PINN", fontsize=16)
    _finish_figure(figure, path)


def plot_nde_evaluation(
    result: NDEEvaluation,
    history: Mapping[str, Sequence[float]],
    path: str | Path,
) -> None:
    plt = _pyplot()

    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    for component, name in enumerate(("angle q", "velocity v")):
        axis = axes[0, component]
        axis.plot(result.times, result.reference[:, component], "k--", label="reference")
        axis.plot(result.times, result.prediction[:, component], label="routed NDE")
        axis.set_title(name)
        axis.set_xlabel("time")
        axis.grid(alpha=0.25)
        axis.legend()

    axes[0, 2].plot(result.times, result.absolute_error[:, 0], label="|q error|")
    axes[0, 2].plot(result.times, result.absolute_error[:, 1], label="|v error|")
    axes[0, 2].set_title("Trajectory error")
    axes[0, 2].set_xlabel("time")
    axes[0, 2].grid(alpha=0.25)
    axes[0, 2].legend()

    axes[1, 0].plot(result.reference[:, 0], result.reference[:, 1], "k--", label="reference")
    axes[1, 0].plot(result.prediction[:, 0], result.prediction[:, 1], label="routed NDE")
    axes[1, 0].set_title("Phase portrait")
    axes[1, 0].set_xlabel("q")
    axes[1, 0].set_ylabel("v")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend()

    route_axis = axes[1, 1]
    route_axis.plot(result.times, result.complex_weight, color="tab:red", label="complex weight")
    route_axis.set_xlabel("time")
    route_axis.set_ylabel("complex weight", color="tab:red")
    route_axis.set_ylim(-0.05, 1.05)
    complexity_axis = route_axis.twinx()
    complexity_axis.plot(result.times, result.complexity, color="tab:blue", alpha=0.6, label="complexity")
    complexity_axis.set_ylabel("Morse complexity", color="tab:blue")
    route_axis.set_title("Routing along rollout")

    loss_axis = axes[1, 2]
    for name, values in history.items():
        if values:
            loss_axis.semilogy(range(1, len(values) + 1), values, label=name)
    loss_axis.set_title("Training losses")
    loss_axis.set_xlabel("epoch")
    loss_axis.set_ylabel("loss")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend()
    figure.suptitle("Morse-routed damped-pendulum NDE", fontsize=16)
    _finish_figure(figure, path)
