"""Complexity-routed heterogeneous PINN and NDE building blocks."""

from .ensemble import HeterogeneousEnsemble, RoutedOutput
from .evaluation import (
    NDEMetrics,
    PINNMetrics,
    evaluate_heat_pinn,
    evaluate_pendulum_nde,
    format_metrics,
    save_metrics,
)
from .experts import FourierExpert, MLPExpert, RBFExpert, build_expert
from .hydrology import (
    HydrologyPINNMambaExpert,
    LearnedHydrologyMorsePotential,
    RoutedHydrologyModel,
    SingleComplexHydrologyModel,
)
from .hydrology_data import load_camels_ch, load_ukraine_csv, make_hydrology_data
from .hydrology_ude import HydrologyUDEData, RoutedHydrologyUDE, SingleComplexHydrologyUDE, make_hydrology_ude_data
from .routing import LearnedRouter, MorseRouter
from .pinnmamba import PINNMamba, PINNMambaExpert, evaluate_reaction_pinnmamba

__all__ = [
    "FourierExpert",
    "HeterogeneousEnsemble",
    "HydrologyPINNMambaExpert",
    "HydrologyUDEData",
    "LearnedHydrologyMorsePotential",
    "MLPExpert",
    "LearnedRouter",
    "MorseRouter",
    "NDEMetrics",
    "PINNMetrics",
    "PINNMamba",
    "PINNMambaExpert",
    "RBFExpert",
    "RoutedHydrologyModel",
    "RoutedHydrologyUDE",
    "RoutedOutput",
    "SingleComplexHydrologyModel",
    "SingleComplexHydrologyUDE",
    "build_expert",
    "evaluate_heat_pinn",
    "evaluate_reaction_pinnmamba",
    "evaluate_pendulum_nde",
    "format_metrics",
    "load_camels_ch",
    "load_ukraine_csv",
    "make_hydrology_data",
    "make_hydrology_ude_data",
    "save_metrics",
]
