from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .experts import MLPExpert, build_expert
from .hydrology_data import HydrologySeries
from .routing import LearnedRouter, MorseRouter


@dataclass(frozen=True)
class HydrologyUDEData:
    forcings: torch.Tensor
    discharge: torch.Tensor
    dates: tuple
    forcing_names: tuple[str, ...]
    forcing_mean: torch.Tensor
    forcing_scale: torch.Tensor
    train_end: int
    validation_end: int
    country: str
    basin_id: str


@dataclass
class UDERollout:
    discharge: torch.Tensor
    states: torch.Tensor
    complex_weight: torch.Tensor
    interface_disagreement: torch.Tensor
    mass_balance_residual: torch.Tensor


def make_hydrology_ude_data(
    series: HydrologySeries,
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> HydrologyUDEData:
    """Create one longest contiguous trajectory and chronological boundaries."""
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must be positive and leave a test period")
    blocks: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(series.dates)):
        if series.dates[index] - series.dates[index - 1] != timedelta(days=1):
            blocks.append((start, index))
            start = index
    blocks.append((start, len(series.dates)))
    first, stop = max(blocks, key=lambda bounds: bounds[1] - bounds[0])
    if stop - first < 30:
        raise ValueError("a contiguous trajectory of at least 30 days is required")
    dates = series.dates[first:stop]
    day = torch.tensor([item.timetuple().tm_yday for item in dates], dtype=torch.float32)
    angle = 2.0 * torch.pi * day / 365.25
    forcings = torch.cat(
        (series.forcings[first:stop], angle.sin()[:, None], angle.cos()[:, None]), dim=1
    )
    discharge = series.discharge[first:stop]
    train_end = int(len(discharge) * train_fraction)
    validation_end = int(len(discharge) * (train_fraction + validation_fraction))
    if min(train_end, validation_end - train_end, len(discharge) - validation_end) < 2:
        raise ValueError("each chronological split must contain at least two days")
    forcing_mean = forcings[:train_end].mean(dim=0)
    forcing_scale = forcings[:train_end].std(dim=0, unbiased=False).clamp_min(1e-6)
    return HydrologyUDEData(
        forcings, discharge, dates, (*series.forcing_names, "day_sin", "day_cos"),
        forcing_mean, forcing_scale, train_end, validation_end,
        series.country, series.basin_id,
    )


def hydrology_state_morse_gradient(inputs: torch.Tensor, precipitation_index: int, pet_index: int) -> torch.Tensor:
    """Gradient of storage energy plus instantaneous climatic water excess."""
    gradient = torch.zeros_like(inputs)
    gradient[:, :3] = inputs[:, :3]
    wetness = inputs[:, precipitation_index] - inputs[:, pet_index]
    gradient[:, precipitation_index] = wetness
    gradient[:, pet_index] = -wetness
    return gradient


class _HydrologyUDEBase(nn.Module):
    state_names = ("snow_storage", "soil_storage", "groundwater_storage")
    correction_names = ("melt", "evapotranspiration", "quickflow", "percolation", "baseflow")

    def __init__(
        self,
        forcing_names: Sequence[str],
        forcing_mean: torch.Tensor,
        forcing_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        self.forcing_names = tuple(forcing_names)
        for required in ("precipitation", "temperature", "pet"):
            if required not in self.forcing_names:
                raise ValueError(f"forcing_names must contain {required!r}")
        self.register_buffer("forcing_mean", forcing_mean.detach().clone().float())
        self.register_buffer("forcing_scale", forcing_scale.detach().clone().float())
        initial = torch.tensor((5.0, 50.0, 10.0))
        self.raw_initial_state = nn.Parameter(torch.log(torch.expm1(initial)))
        self.base_flux_logits = nn.Parameter(torch.tensor((-1.0, -0.5, -2.0, -2.5, -2.5)))
        self.correction_bound = 1.5

    @property
    def input_dim(self) -> int:
        return 3 + len(self.forcing_names)

    def initial_state(self, batch_size: int = 1) -> torch.Tensor:
        return F.softplus(self.raw_initial_state)[None, :].expand(batch_size, -1)

    def field_inputs(self, state: torch.Tensor, forcing: torch.Tensor) -> torch.Tensor:
        positive_state = state.clamp_min(0.0)
        state_scale = torch.tensor((20.0, 100.0, 50.0), device=state.device, dtype=state.dtype)
        state_features = torch.log1p(positive_state) / torch.log1p(state_scale)
        normalized_forcing = (forcing - self.forcing_mean) / self.forcing_scale
        return torch.cat((state_features, normalized_forcing), dim=-1)

    def correction(
        self, inputs: torch.Tensor, *, physics_only: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def vector_field(
        self,
        state: torch.Tensor,
        forcing: torch.Tensor,
        *,
        physics_only: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        positive_state = state.clamp_min(0.0)
        snow, soil, groundwater = positive_state.unbind(dim=-1)
        precipitation = forcing[:, self.forcing_names.index("precipitation")].clamp_min(0.0)
        temperature = forcing[:, self.forcing_names.index("temperature")]
        pet = forcing[:, self.forcing_names.index("pet")].clamp_min(0.0)
        inputs = self.field_inputs(state, forcing)
        correction, weight, disagreement = self.correction(inputs, physics_only=physics_only)
        logits = self.base_flux_logits[None, :] + self.correction_bound * torch.tanh(correction)

        snow_fraction = torch.sigmoid(-temperature / 1.5)
        snowfall = precipitation * snow_fraction
        rainfall = precipitation - snowfall
        melt = snow * torch.sigmoid(logits[:, 0]) * torch.sigmoid(temperature / 2.0)
        evapotranspiration = (
            soil * torch.sigmoid(logits[:, 1])
            * (1.0 - torch.exp(-pet / (soil + 1e-6)))
        )
        allocation = torch.softmax(
            torch.stack((torch.zeros_like(soil), logits[:, 2], logits[:, 3]), dim=-1), dim=-1
        )
        quickflow = soil * allocation[:, 1]
        percolation = soil * allocation[:, 2]
        baseflow = groundwater * torch.sigmoid(logits[:, 4])
        discharge = quickflow + baseflow
        derivative = torch.stack(
            (
                snowfall - melt,
                rainfall + melt - evapotranspiration - quickflow - percolation,
                percolation - baseflow,
            ),
            dim=-1,
        )
        return derivative, discharge, evapotranspiration, weight, disagreement, inputs

    def step(
        self, state: torch.Tensor, forcing: torch.Tensor, *, physics_only: bool = False
    ) -> tuple[torch.Tensor, ...]:
        first = self.vector_field(state, forcing, physics_only=physics_only)
        second = self.vector_field(state + first[0], forcing, physics_only=physics_only)

        def average(index: int) -> torch.Tensor:
            return 0.5 * (first[index] + second[index])

        derivative = average(0)
        discharge = average(1)
        evapotranspiration = average(2)
        complex_weight = average(3)
        disagreement = average(4)
        next_state = state + derivative
        precipitation = forcing[:, self.forcing_names.index("precipitation")].clamp_min(0.0)
        mass_residual = (
            next_state.sum(dim=-1) - state.sum(dim=-1)
            - (precipitation - evapotranspiration - discharge)
        )
        return next_state, discharge, complex_weight, disagreement, mass_residual

    def rollout(
        self,
        forcings: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        *,
        physics_only: bool = False,
    ) -> UDERollout:
        squeeze = forcings.ndim == 2
        if squeeze:
            forcings = forcings.unsqueeze(0)
        if forcings.ndim != 3 or forcings.shape[-1] != len(self.forcing_names):
            raise ValueError("forcings must have shape [time, features] or [batch, time, features]")
        state = self.initial_state(len(forcings)) if initial_state is None else initial_state
        discharges, states, weights, disagreements, mass_residuals = [], [], [], [], []
        for index in range(forcings.shape[1]):
            state, discharge, weight, disagreement, mass_residual = self.step(
                state, forcings[:, index], physics_only=physics_only
            )
            discharges.append(discharge)
            states.append(state)
            weights.append(weight)
            disagreements.append(disagreement)
            mass_residuals.append(mass_residual)
        result = UDERollout(
            torch.stack(discharges, dim=1), torch.stack(states, dim=1),
            torch.stack(weights, dim=1), torch.stack(disagreements, dim=1),
            torch.stack(mass_residuals, dim=1),
        )
        if squeeze:
            return UDERollout(*(value.squeeze(0) for value in result.__dict__.values()))
        return result

    @torch.no_grad()
    def physics_reference_inputs(self, forcings: torch.Tensor) -> torch.Tensor:
        squeeze = forcings.ndim == 2
        if squeeze:
            forcings = forcings.unsqueeze(0)
        state = self.initial_state(len(forcings))
        references = []
        for index in range(forcings.shape[1]):
            forcing = forcings[:, index]
            references.append(self.field_inputs(state, forcing))
            state = self.step(state, forcing, physics_only=True)[0]
        return torch.cat(references, dim=0)


class RoutedHydrologyUDE(_HydrologyUDEBase):
    """Universal differential equation with routed expert flux corrections."""

    def __init__(
        self,
        forcing_names: Sequence[str],
        forcing_mean: torch.Tensor,
        forcing_scale: torch.Tensor,
        *,
        simple_kind: str = "rbf",
        routing: str = "morse",
        complexity_percentile: float = 80.0,
        gate_temperature: float = 0.15,
    ) -> None:
        super().__init__(forcing_names, forcing_mean, forcing_scale)
        simple_kwargs: dict[str, object] = {}
        if simple_kind == "rbf":
            simple_kwargs = {
                "n_centers": 32,
                "domain_low": [-3.0] * self.input_dim,
                "domain_high": [3.0] * self.input_dim,
                "initial_width": math.sqrt(self.input_dim),
            }
        elif simple_kind == "fourier":
            simple_kwargs = {"n_frequencies": 32, "frequency_scale": 1.0 / math.sqrt(self.input_dim)}
        self.simple_expert = build_expert(simple_kind, self.input_dim, 5, **simple_kwargs)
        self.complex_expert = MLPExpert(self.input_dim, 5, hidden_dims=(64, 64))
        if routing == "morse":
            precipitation = 3 + self.forcing_names.index("precipitation")
            pet = 3 + self.forcing_names.index("pet")
            self.router: MorseRouter | LearnedRouter = MorseRouter(
                lambda values: hydrology_state_morse_gradient(values, precipitation, pet),
                complexity_percentile, gate_temperature,
            )
        elif routing == "learned":
            self.router = LearnedRouter(
                self.input_dim, target_complex_fraction=1.0 - complexity_percentile / 100.0
            )
        else:
            raise ValueError("routing must be 'morse' or 'learned'")
        self.simple_kind = simple_kind
        self.routing_kind = routing

    def fit_router(self, forcings: torch.Tensor) -> "RoutedHydrologyUDE":
        if self.routing_kind == "morse":
            self.router.fit(self.physics_reference_inputs(forcings))
        return self

    def correction(
        self, inputs: torch.Tensor, *, physics_only: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if physics_only:
            zero = torch.zeros((len(inputs), 5), device=inputs.device, dtype=inputs.dtype)
            return zero, torch.zeros(len(inputs), device=inputs.device, dtype=inputs.dtype), torch.zeros(len(inputs), device=inputs.device, dtype=inputs.dtype)
        weight = self.router.complex_weight(inputs)
        simple = self.simple_expert(inputs)
        complex_value = self.complex_expert(inputs)
        correction = torch.lerp(simple, complex_value, weight[:, None])
        transition = 4.0 * weight * (1.0 - weight)
        disagreement = transition * (simple - complex_value).square().mean(dim=-1)
        return correction, weight, disagreement


class SingleComplexHydrologyUDE(_HydrologyUDEBase):
    """Physics plus the same MLP correction without routing."""

    def __init__(
        self,
        forcing_names: Sequence[str],
        forcing_mean: torch.Tensor,
        forcing_scale: torch.Tensor,
    ) -> None:
        super().__init__(forcing_names, forcing_mean, forcing_scale)
        self.complex_expert = MLPExpert(self.input_dim, 5, hidden_dims=(64, 64))

    def correction(
        self, inputs: torch.Tensor, *, physics_only: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        correction = torch.zeros((len(inputs), 5), device=inputs.device, dtype=inputs.dtype) if physics_only else self.complex_expert(inputs)
        one = torch.ones(len(inputs), device=inputs.device, dtype=inputs.dtype)
        return correction, one, torch.zeros_like(one)
