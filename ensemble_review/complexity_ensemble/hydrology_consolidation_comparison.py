from __future__ import annotations

from dataclasses import dataclass

import torch

from .hydrology import (
    LearnedHydrologyMorsePotential,
    RoutedHydrologyModel,
    SingleComplexHydrologyModel,
    SingleSimpleHydrologyModel,
)
from .hydrology_comparison import (
    _add_observation_noise,
    _evaluate,
    _make_model,
    _parameter_count,
    hydrology_comparison_summary,
    plot_hydrology_comparison,
    save_hydrology_comparison,
)
from .hydrology_data import HydrologyData, HydrologySplit


@dataclass(frozen=True)
class HydrologyConsolidationComparisonRecord:
    country: str
    basin: str
    model: str
    simple_expert: str
    complex_expert: str
    seed: int
    training_noise: float
    inference_noise: float
    validation_nse: float
    test_nse: float
    test_kge: float
    test_rmse_mm_day: float
    physics_error: float
    parameters: int
    mean_complex_weight: float
    consolidation_weight: float


def train_hydrology_model_with_consolidation(
    model: RoutedHydrologyModel | SingleComplexHydrologyModel | SingleSimpleHydrologyModel,
    split: HydrologySplit,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    training_noise: float,
    seed: int,
    device: torch.device,
    consolidation_weight: float,
    feature_names: tuple[str, ...],
) -> None:
    """Train without changing the behavior of the standard hydrology trainer."""
    if consolidation_weight < 0.0:
        raise ValueError("consolidation_weight must be non-negative")
    if training_noise < 0.0:
        raise ValueError("training_noise must be non-negative")
    inputs = split.inputs.to(device)
    physical = split.physical_inputs.to(device)
    targets = split.targets.to(device)
    if isinstance(model, RoutedHydrologyModel):
        model.fit_router(inputs)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device=device).manual_seed(seed + 104729)
    for _ in range(epochs):
        permutation = torch.randperm(len(inputs), generator=generator, device=device)
        for start in range(0, len(inputs), batch_size):
            indices = permutation[start : start + batch_size]
            batch_inputs = _add_observation_noise(
                inputs[indices], feature_names, training_noise, generator
            )
            optimizer.zero_grad()
            if isinstance(model, RoutedHydrologyModel):
                losses = model.losses(
                    batch_inputs,
                    physical[indices],
                    targets[indices],
                    interface_weight=consolidation_weight,
                )
            else:
                losses = model.losses(batch_inputs, physical[indices], targets[indices])
            losses.total.backward()
            optimizer.step()


def compare_hydrology_models_with_consolidation(
    data: HydrologyData,
    *,
    models: tuple[str, ...] | None = None,
    simple_kind: str = "rbf",
    complex_kind: str = "mlp",
    seeds: tuple[int, ...] = (0,),
    training_noise_levels: tuple[float, ...] = (0.0,),
    inference_noise_levels: tuple[float, ...] = (0.0, 0.05),
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    consolidation_weight: float = 0.05,
    device: torch.device | str = "cpu",
    learned_morse_potential: LearnedHydrologyMorsePotential | None = None,
) -> list[HydrologyConsolidationComparisonRecord]:
    """Run an isolated comparison with configurable boundary consolidation."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if consolidation_weight < 0.0:
        raise ValueError("consolidation_weight must be non-negative")
    if any(noise < 0.0 for noise in (*training_noise_levels, *inference_noise_levels)):
        raise ValueError("noise levels must be non-negative")
    available_models = {"morse", "learned", "single_complex"}
    if learned_morse_potential is not None:
        available_models.add("learned_morse")
    if models is None:
        selected_models = ["morse"]
        if learned_morse_potential is not None:
            selected_models.append("learned_morse")
        selected_models.extend(("learned", "single_complex"))
    else:
        selected_models = list(dict.fromkeys(models))
        invalid_models = set(selected_models) - available_models
        if invalid_models:
            choices = ", ".join(sorted(available_models))
            invalid = ", ".join(sorted(invalid_models))
            raise ValueError(f"unknown or unavailable models: {invalid}; choose from {choices}")
        if not selected_models:
            raise ValueError("models must not be empty")

    device = torch.device(device)
    records = []
    for seed in seeds:
        for training_noise in sorted(set(training_noise_levels)):
            for model_name in selected_models:
                torch.manual_seed(seed)
                model = _make_model(
                    model_name, data, simple_kind, complex_kind, learned_morse_potential
                ).to(device)
                train_hydrology_model_with_consolidation(
                    model,
                    data.train,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    training_noise=training_noise,
                    seed=seed,
                    device=device,
                    consolidation_weight=consolidation_weight,
                    feature_names=data.feature_names,
                )
                for inference_noise in sorted(set(inference_noise_levels)):
                    validation_nse, _, _, _, _ = _evaluate(
                        model,
                        data.validation,
                        data,
                        inference_noise=inference_noise,
                        seed=seed + 200003,
                        device=device,
                    )
                    test_nse, test_kge, test_rmse, physics, usage = _evaluate(
                        model,
                        data.test,
                        data,
                        inference_noise=inference_noise,
                        seed=seed + 300007,
                        device=device,
                    )
                    records.append(
                        HydrologyConsolidationComparisonRecord(
                            data.country,
                            data.basin_id,
                            model_name,
                            simple_kind if model_name != "single_complex" else "none",
                            complex_kind,
                            seed,
                            training_noise,
                            inference_noise,
                            validation_nse,
                            test_nse,
                            test_kge,
                            test_rmse,
                            physics,
                            _parameter_count(model),
                            usage,
                            consolidation_weight,
                        )
                    )
    return records


__all__ = [
    "HydrologyConsolidationComparisonRecord",
    "compare_hydrology_models_with_consolidation",
    "hydrology_comparison_summary",
    "plot_hydrology_comparison",
    "save_hydrology_comparison",
    "train_hydrology_model_with_consolidation",
]
