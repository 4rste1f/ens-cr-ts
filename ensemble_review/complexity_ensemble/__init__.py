"""Complexity-routed hydrology ensemble building blocks."""

from .experts import FourierExpert, MLPExpert, RBFExpert, build_expert
from .hydrology import (
    HydrologyPINNMambaExpert,
    LearnedHydrologyMorsePotential,
    RoutedHydrologyModel,
    SingleComplexHydrologyModel,
)
from .hydrology_data import load_camels_ch, load_ukraine_csv, make_hydrology_data
from .hydrology_extreme_comparison import (
    ENSEMBLE_APPROACHES,
    ExtremeComparisonConfig,
    ExtremeEnsembleRecord,
    compare_extreme_event_ensembles,
)
from .hydrology_complexity import (
    HydrologyComplexityEstimator,
    LyapunovComplexityEstimator,
    TakensPersistenceEstimator,
    build_hydrology_complexity_estimator,
    takens_delay_embedding,
)
from .routing import LearnedRouter, MorseRouter, ScoreRouter
from .pinnmamba import PINNMamba, PINNMambaExpert, evaluate_reaction_pinnmamba
from .regional import (
    BasinScopeConfig,
    CAMELSCHCatalog,
    CancellationToken,
    DateRange,
    DateSplitConfig,
    ExperimentResult,
    ExtremeEventConfig,
    HyperparameterConfig,
    ModelArchitectureConfig,
    PhysicsModelConfig,
    RegionalExperimentConfig,
    TrainingStrategyConfig,
    make_regional_hydrology_data,
    run_regional_experiment,
    write_result_artifacts,
)

__all__ = [
    "FourierExpert",
    "ENSEMBLE_APPROACHES",
    "ExtremeComparisonConfig",
    "ExtremeEnsembleRecord",
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
    "compare_extreme_event_ensembles",
    "evaluate_reaction_pinnmamba",
    "load_camels_ch",
    "load_ukraine_csv",
    "make_hydrology_data",
    "make_regional_hydrology_data",
    "BasinScopeConfig",
    "CAMELSCHCatalog",
    "CancellationToken",
    "DateRange",
    "DateSplitConfig",
    "ExperimentResult",
    "ExtremeEventConfig",
    "HyperparameterConfig",
    "ModelArchitectureConfig",
    "PhysicsModelConfig",
    "RegionalExperimentConfig",
    "TrainingStrategyConfig",
    "run_regional_experiment",
    "write_result_artifacts",
    "takens_delay_embedding",
]
