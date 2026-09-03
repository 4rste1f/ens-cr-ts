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
from .hydrology_complexity import (
    HydrologyComplexityEstimator,
    LyapunovComplexityEstimator,
    TakensPersistenceEstimator,
    build_hydrology_complexity_estimator,
    takens_delay_embedding,
)
from .hydrology_ude import HydrologyUDEData, RoutedHydrologyUDE, SingleComplexHydrologyUDE, make_hydrology_ude_data
from .routing import LearnedRouter, MorseRouter, ScoreRouter
from .pinnmamba import PINNMamba, PINNMambaExpert, evaluate_reaction_pinnmamba

__all__ = [
    "FourierExpert",
    "HeterogeneousEnsemble",
    "HydrologyPINNMambaExpert",
    "HydrologyComplexityEstimator",
    "HydrologyUDEData",
    "LearnedHydrologyMorsePotential",
    "MLPExpert",
    "LearnedRouter",
    "MorseRouter",
    "LyapunovComplexityEstimator",
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
    "ScoreRouter",
    "TakensPersistenceEstimator",
    "build_expert",
    "build_hydrology_complexity_estimator",
    "evaluate_heat_pinn",
    "evaluate_reaction_pinnmamba",
    "evaluate_pendulum_nde",
    "format_metrics",
    "load_camels_ch",
    "load_ukraine_csv",
    "make_hydrology_data",
    "make_hydrology_ude_data",
    "save_metrics",
    "takens_delay_embedding",
]
