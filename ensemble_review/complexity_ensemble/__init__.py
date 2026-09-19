"""Complexity-routed hydrology ensemble building blocks."""

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
from .routing import LearnedRouter, MorseRouter, ScoreRouter
from .pinnmamba import PINNMamba, PINNMambaExpert, evaluate_reaction_pinnmamba

__all__ = [
    "FourierExpert",
    "HydrologyPINNMambaExpert",
    "HydrologyComplexityEstimator",
    "LearnedHydrologyMorsePotential",
    "MLPExpert",
    "LearnedRouter",
    "MorseRouter",
    "LyapunovComplexityEstimator",
    "PINNMamba",
    "PINNMambaExpert",
    "RBFExpert",
    "RoutedHydrologyModel",
    "SingleComplexHydrologyModel",
    "ScoreRouter",
    "TakensPersistenceEstimator",
    "build_expert",
    "build_hydrology_complexity_estimator",
    "evaluate_reaction_pinnmamba",
    "load_camels_ch",
    "load_ukraine_csv",
    "make_hydrology_data",
    "takens_delay_embedding",
]
