from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .ensemble import HeterogeneousEnsemble, RoutedOutput
from .experts import MLPExpert, build_expert
from .pinnmamba import PINNMambaExpert
from .routing import LearnedRouter, MorseRouter, heat_morse_gradient


@dataclass
class PINNLosses:
    total: torch.Tensor
    initial: torch.Tensor
    boundary: torch.Tensor
    physics: torch.Tensor
    interface: torch.Tensor
    complex_usage: torch.Tensor
    routing: torch.Tensor
    alignment: torch.Tensor


class HeatPINN(nn.Module):
    """Morse-routed PINN for ``u_t = diffusivity * u_xx`` on [-1, 1]."""

    def __init__(
        self,
        simple_kind: str = "rbf",
        diffusivity: float = 0.05,
        complexity_percentile: float = 80.0,
        gate_temperature: float = 0.15,
        simple_kwargs: dict[str, object] | None = None,
        complex_kwargs: dict[str, object] | None = None,
        routing: str = "morse",
        complex_kind: str = "mlp",
        sequence_length: int = 7,
        sequence_step: float = 0.01,
        physics_token_samples: int = 1,
    ) -> None:
        super().__init__()
        simple_kwargs = dict(simple_kwargs or {})
        complex_kwargs = dict(complex_kwargs or {})
        if simple_kind == "rbf":
            simple_kwargs.setdefault("domain_low", [0.0, -1.0])
            simple_kwargs.setdefault("domain_high", [1.0, 1.0])
        simple = build_expert(simple_kind, 2, 1, **simple_kwargs)
        if complex_kind == "mlp":
            complex_expert = MLPExpert(2, 1, **complex_kwargs)
        elif complex_kind == "pinnmamba":
            complex_expert = PINNMambaExpert(
                sequence_length=sequence_length,
                sequence_step=sequence_step,
                **complex_kwargs,
            )
        else:
            raise ValueError("complex_kind must be 'mlp' or 'pinnmamba'")
        if routing == "morse":
            router = MorseRouter(heat_morse_gradient, complexity_percentile, gate_temperature)
        elif routing == "learned":
            router = LearnedRouter(2, target_complex_fraction=1.0 - complexity_percentile / 100.0)
        else:
            raise ValueError("routing must be 'morse' or 'learned'")
        self.ensemble = HeterogeneousEnsemble(simple, complex_expert, router)
        self.diffusivity = diffusivity
        self.complex_kind = complex_kind
        if not 1 <= physics_token_samples <= sequence_length:
            raise ValueError("physics_token_samples must be between 1 and sequence_length")
        self.physics_token_samples = physics_token_samples

    @property
    def router(self) -> MorseRouter:
        return self.ensemble.router

    def routing_reference_points(self, tx: torch.Tensor) -> torch.Tensor:
        """Include sequence tokens when calibrating a PINNMamba routing gate."""
        expert = self.ensemble.complex_expert
        if isinstance(expert, PINNMambaExpert):
            return expert.make_tx_sequence(tx).reshape(-1, 2)
        return tx

    def fit_router(self, tx: torch.Tensor) -> "HeatPINN":
        self.router.fit(self.routing_reference_points(tx))
        return self

    def forward(
        self, tx: torch.Tensor, *, hard: bool = False, return_details: bool = False
    ) -> torch.Tensor | RoutedOutput:
        return self.ensemble(tx, hard=hard, return_details=return_details)

    def residual(self, tx: torch.Tensor) -> torch.Tensor:
        """Evaluate the PDE on the final, smoothly routed solution everywhere."""
        tx = tx.detach().clone().requires_grad_(True)
        prediction = self(tx)
        assert isinstance(prediction, torch.Tensor)
        gradient = torch.autograd.grad(
            prediction, tx, torch.ones_like(prediction), create_graph=True
        )[0]
        u_t = gradient[:, 0:1]
        u_x = gradient[:, 1:2]
        u_xx = torch.autograd.grad(u_x, tx, torch.ones_like(u_x), create_graph=True)[0][:, 1:2]
        return u_t - self.diffusivity * u_xx

    def _sequence_output(
        self, tx_sequence: torch.Tensor, *, return_details: bool = False
    ) -> torch.Tensor | RoutedOutput:
        if self.complex_kind != "pinnmamba":
            raise RuntimeError("sequence output requires the PINNMamba expert")
        expert = self.ensemble.complex_expert
        assert isinstance(expert, PINNMambaExpert)
        flat = tx_sequence.reshape(-1, 2)
        weight = self.router.complex_weight(flat).reshape(tx_sequence.shape[:-1])
        simple = self.ensemble.simple_expert(flat).reshape(*tx_sequence.shape[:-1], -1)
        complex_value = expert.forward_tx_sequence(tx_sequence)
        value = torch.lerp(simple, complex_value, weight.unsqueeze(-1))
        if return_details:
            return RoutedOutput(value, weight, simple, complex_value)
        return value

    def sequence_residual(self, tx: torch.Tensor, *, all_tokens: bool = False) -> torch.Tensor:
        """Evaluate exact diagonal PDE derivatives at sampled sequence tokens.

        Sampling tokens gives an unbiased estimate of the full sequence physics
        loss and avoids materializing a large second-order recurrent Jacobian.
        Set ``all_tokens=True`` for diagnostics on small batches.
        """
        expert = self.ensemble.complex_expert
        if not isinstance(expert, PINNMambaExpert):
            return self.residual(tx)
        sequence = expert.make_tx_sequence(tx).detach().clone().requires_grad_(True)
        prediction = self._sequence_output(sequence)
        assert isinstance(prediction, torch.Tensor)
        # A recurrent output at token k also depends on earlier input tokens.
        # A single vector-Jacobian product would sum those cross-token terms.
        # The PDE instead needs the diagonal derivatives du_k/d(t_k,x_k).
        if all_tokens or self.physics_token_samples == prediction.shape[1]:
            token_indices = torch.arange(prediction.shape[1], device=prediction.device)
        else:
            token_indices = torch.randperm(prediction.shape[1], device=prediction.device)[
                : self.physics_token_samples
            ]
        u_t_tokens = []
        u_x_tokens = []
        for token_tensor in token_indices:
            token = int(token_tensor)
            token_gradient = torch.autograd.grad(
                prediction[:, token].sum(),
                sequence,
                retain_graph=True,
                create_graph=True,
            )[0][:, token]
            u_t_tokens.append(token_gradient[:, 0:1])
            u_x_tokens.append(token_gradient[:, 1:2])
        u_t = torch.stack(u_t_tokens, dim=1)
        u_x = torch.stack(u_x_tokens, dim=1)
        u_xx_tokens = []
        for output_index, token_tensor in enumerate(token_indices):
            token = int(token_tensor)
            second_gradient = torch.autograd.grad(
                u_x[:, output_index].sum(),
                sequence,
                retain_graph=True,
                create_graph=True,
            )[0][:, token, 1:2]
            u_xx_tokens.append(second_gradient)
        u_xx = torch.stack(u_xx_tokens, dim=1)
        return u_t - self.diffusivity * u_xx

    def sequence_alignment_loss(self, tx: torch.Tensor) -> torch.Tensor:
        """Align assembled predictions from two overlapping time windows."""
        expert = self.ensemble.complex_expert
        if not isinstance(expert, PINNMambaExpert):
            return torch.zeros((), device=tx.device, dtype=tx.dtype)
        first = self._sequence_output(expert.make_tx_sequence(tx))
        shifted = tx.clone()
        shifted[:, 0] = shifted[:, 0] + expert.sequence_step
        second = self._sequence_output(expert.make_tx_sequence(shifted))
        assert isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor)
        return (first[:, 1:] - second[:, :-1]).square().mean()

    def sequence_interface_loss(self, tx: torch.Tensor) -> torch.Tensor:
        expert = self.ensemble.complex_expert
        if not isinstance(expert, PINNMambaExpert):
            return self.ensemble.interface_loss(tx)
        routed = self._sequence_output(expert.make_tx_sequence(tx), return_details=True)
        assert isinstance(routed, RoutedOutput)
        transition = 4.0 * routed.complex_weight * (1.0 - routed.complex_weight)
        disagreement = (routed.simple_value - routed.complex_value).square().mean(dim=-1)
        return (transition * disagreement).mean()

    def losses(
        self,
        initial_tx: torch.Tensor,
        initial_u: torch.Tensor,
        boundary_tx: torch.Tensor,
        boundary_u: torch.Tensor,
        collocation_tx: torch.Tensor,
        *,
        physics_weight: float = 1.0,
        interface_weight: float = 0.05,
        compute_weight: float = 0.0,
        routing_weight: float = 0.05,
        alignment_weight: float = 1000.0,
    ) -> PINNLosses:
        initial = (self(initial_tx) - initial_u).square().mean()
        boundary = (self(boundary_tx) - boundary_u).square().mean()
        physics = self.sequence_residual(collocation_tx).square().mean()
        interface = self.sequence_interface_loss(collocation_tx)
        if self.complex_kind == "pinnmamba":
            expert = self.ensemble.complex_expert
            assert isinstance(expert, PINNMambaExpert)
            routing_points = expert.make_tx_sequence(collocation_tx).reshape(-1, 2)
            alignment = self.sequence_alignment_loss(collocation_tx)
        else:
            routing_points = collocation_tx
            alignment = torch.zeros((), device=collocation_tx.device, dtype=collocation_tx.dtype)
        complex_usage = self.router.complex_weight(routing_points).mean()
        routing = self.router.regularization(routing_points)
        total = (
            initial
            + boundary
            + physics_weight * physics
            + interface_weight * interface
            + compute_weight * complex_usage
            + routing_weight * routing
            + alignment_weight * alignment
        )
        return PINNLosses(total, initial, boundary, physics, interface, complex_usage, routing, alignment)


def sample_heat_problem(
    n_initial: int = 128,
    n_boundary: int = 128,
    n_collocation: int = 1024,
    *,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, ...]:
    device = torch.device(device)
    x_initial = 2.0 * torch.rand(n_initial, 1, device=device) - 1.0
    initial_tx = torch.cat((torch.zeros_like(x_initial), x_initial), dim=1)
    initial_u = torch.sin(torch.pi * x_initial)

    t_boundary = torch.rand(n_boundary, 1, device=device)
    signs = torch.where(
        torch.arange(n_boundary, device=device)[:, None] % 2 == 0,
        -torch.ones_like(t_boundary),
        torch.ones_like(t_boundary),
    )
    boundary_tx = torch.cat((t_boundary, signs), dim=1)
    boundary_u = torch.zeros(n_boundary, 1, device=device)

    collocation_tx = torch.rand(n_collocation, 2, device=device)
    collocation_tx[:, 1] = 2.0 * collocation_tx[:, 1] - 1.0
    return initial_tx, initial_u, boundary_tx, boundary_u, collocation_tx


def heat_solution(tx: torch.Tensor, diffusivity: float = 0.05) -> torch.Tensor:
    t, x = tx[:, 0:1], tx[:, 1:2]
    return torch.exp(-diffusivity * torch.pi**2 * t) * torch.sin(torch.pi * x)
