from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .ensemble import HeterogeneousEnsemble, RoutedOutput
from .experts import MLPExpert, build_expert
from .routing import LearnedRouter, MorseRouter, pendulum_morse_gradient


@dataclass
class NDELosses:
    total: torch.Tensor
    data: torch.Tensor
    kinematic_physics: torch.Tensor
    interface: torch.Tensor
    complex_usage: torch.Tensor
    routing: torch.Tensor


class MorseRoutedVectorField(nn.Module):
    """History-independent routed vector field with an SSM-ready expert boundary."""

    def __init__(
        self,
        state_dim: int,
        simple_kind: str = "rbf",
        complexity_percentile: float = 80.0,
        gate_temperature: float = 0.15,
        simple_kwargs: dict[str, object] | None = None,
        complex_kwargs: dict[str, object] | None = None,
        routing: str = "morse",
    ) -> None:
        super().__init__()
        input_dim = state_dim + 1
        simple_kwargs = dict(simple_kwargs or {})
        complex_kwargs = dict(complex_kwargs or {})
        if simple_kind == "rbf":
            simple_kwargs.setdefault("domain_low", [-1.0, *([-3.0] * state_dim)])
            simple_kwargs.setdefault("domain_high", [20.0, *([3.0] * state_dim)])
        simple = build_expert(simple_kind, input_dim, state_dim, **simple_kwargs)
        complex_expert = MLPExpert(input_dim, state_dim, **complex_kwargs)
        if routing == "morse":
            router = MorseRouter(pendulum_morse_gradient, complexity_percentile, gate_temperature)
        elif routing == "learned":
            router = LearnedRouter(input_dim, target_complex_fraction=1.0 - complexity_percentile / 100.0)
        else:
            raise ValueError("routing must be 'morse' or 'learned'")
        self.ensemble = HeterogeneousEnsemble(simple, complex_expert, router)
        self.state_dim = state_dim

    @property
    def router(self) -> MorseRouter:
        return self.ensemble.router

    @staticmethod
    def with_time(t: torch.Tensor | float, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=state.dtype, device=state.device)
        t = t.to(dtype=state.dtype, device=state.device)
        if t.ndim == 0:
            t = t.expand(len(state), 1)
        elif t.ndim == 1:
            t = t.unsqueeze(-1)
        return torch.cat((t, state), dim=-1)

    def forward(
        self,
        t: torch.Tensor | float,
        state: torch.Tensor,
        *,
        hard: bool = False,
        return_details: bool = False,
    ) -> torch.Tensor | RoutedOutput:
        squeeze = state.ndim == 1
        inputs = self.with_time(t, state)
        result = self.ensemble(inputs, hard=hard, return_details=return_details)
        if squeeze and isinstance(result, torch.Tensor):
            return result.squeeze(0)
        return result

    def losses(
        self,
        inputs: torch.Tensor,
        target_derivatives: torch.Tensor,
        *,
        physics_weight: float = 0.1,
        interface_weight: float = 0.05,
        compute_weight: float = 0.0,
        routing_weight: float = 0.05,
    ) -> NDELosses:
        routed = self.ensemble(inputs, return_details=True)
        assert isinstance(routed, RoutedOutput)
        data = (routed.value - target_derivatives).square().mean()
        # For q'=v, enforce known kinematics on the assembled vector field.
        kinematic = (routed.value[:, 0] - inputs[:, 2]).square().mean()
        interface = self.ensemble.interface_loss(inputs)
        complex_usage = routed.complex_weight.mean()
        routing = self.router.regularization(inputs)
        total = (
            data
            + physics_weight * kinematic
            + interface_weight * interface
            + compute_weight * complex_usage
            + routing_weight * routing
        )
        return NDELosses(total, data, kinematic, interface, complex_usage, routing)


def rk4_step(
    vector_field: nn.Module, t: torch.Tensor, state: torch.Tensor, step_size: torch.Tensor
) -> torch.Tensor:
    k1 = vector_field(t, state)
    k2 = vector_field(t + step_size / 2, state + step_size * k1 / 2)
    k3 = vector_field(t + step_size / 2, state + step_size * k2 / 2)
    k4 = vector_field(t + step_size, state + step_size * k3)
    return state + step_size * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def integrate_rk4(vector_field: nn.Module, initial_state: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    """Differentiable fixed-step integration; adaptive-solver routing comes later."""
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("times must be a one-dimensional tensor with at least two entries")
    states = [initial_state]
    state = initial_state
    for left, right in zip(times[:-1], times[1:]):
        state = rk4_step(vector_field, left, state, right - left)
        states.append(state)
    return torch.stack(states)


class DampedPendulum:
    """Non-polynomial reference dynamics used to avoid an exactly specified simple expert."""

    def __init__(self, damping: float = 0.15, forcing: float = 0.25, frequency: float = 1.2) -> None:
        self.damping = damping
        self.forcing = forcing
        self.frequency = frequency

    def __call__(self, t: torch.Tensor | float, state: torch.Tensor) -> torch.Tensor:
        q, velocity = state[..., 0], state[..., 1]
        t_tensor = torch.as_tensor(t, dtype=state.dtype, device=state.device)
        acceleration = (
            -torch.sin(q)
            - self.damping * velocity
            + self.forcing * torch.cos(self.frequency * t_tensor)
        )
        return torch.stack((velocity, acceleration), dim=-1)


def sample_pendulum_derivatives(
    n_samples: int = 2048, *, device: torch.device | str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device(device)
    time = 20.0 * torch.rand(n_samples, 1, device=device)
    state = 6.0 * torch.rand(n_samples, 2, device=device) - 3.0
    inputs = torch.cat((time, state), dim=-1)
    target = DampedPendulum()(time[:, 0], state)
    return inputs, target
