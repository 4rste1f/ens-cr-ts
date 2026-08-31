from __future__ import annotations

import torch
from torch import nn

from .experts import MLPExpert
from .pinnmamba import PINNMambaExpert


class SingleComplexHeatPINN(nn.Module):
    """Unitary heat baseline using the selected complex-expert architecture."""

    def __init__(
        self,
        complex_kind: str = "mlp",
        diffusivity: float = 0.05,
        complex_kwargs: dict[str, object] | None = None,
        sequence_length: int = 7,
        sequence_step: float = 0.01,
        physics_token_samples: int = 1,
    ) -> None:
        super().__init__()
        complex_kwargs = dict(complex_kwargs or {})
        if complex_kind == "mlp":
            self.net: nn.Module = MLPExpert(2, 1, **complex_kwargs)
        elif complex_kind == "pinnmamba":
            self.net = PINNMambaExpert(
                sequence_length=sequence_length,
                sequence_step=sequence_step,
                **complex_kwargs,
            )
        else:
            raise ValueError("complex_kind must be 'mlp' or 'pinnmamba'")
        if not 1 <= physics_token_samples <= sequence_length:
            raise ValueError("physics_token_samples must be between 1 and sequence_length")
        self.complex_kind = complex_kind
        self.diffusivity = diffusivity
        self.physics_token_samples = physics_token_samples

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

    def residual(self, tx: torch.Tensor) -> torch.Tensor:
        tx = tx.detach().clone().requires_grad_(True)
        prediction = self(tx)
        gradient = torch.autograd.grad(prediction, tx, torch.ones_like(prediction), create_graph=True)[0]
        u_t, u_x = gradient[:, 0:1], gradient[:, 1:2]
        u_xx = torch.autograd.grad(u_x, tx, torch.ones_like(u_x), create_graph=True)[0][:, 1:2]
        return u_t - self.diffusivity * u_xx

    def sequence_residual(self, tx: torch.Tensor) -> torch.Tensor:
        if not isinstance(self.net, PINNMambaExpert):
            return self.residual(tx)
        sequence = self.net.make_tx_sequence(tx).detach().clone().requires_grad_(True)
        prediction = self.net.forward_tx_sequence(sequence)
        token_indices = torch.randperm(prediction.shape[1], device=prediction.device)[
            : self.physics_token_samples
        ]
        u_t_tokens = []
        u_x_tokens = []
        for token_tensor in token_indices:
            token = int(token_tensor)
            gradient = torch.autograd.grad(
                prediction[:, token].sum(), sequence, retain_graph=True, create_graph=True
            )[0][:, token]
            u_t_tokens.append(gradient[:, 0:1])
            u_x_tokens.append(gradient[:, 1:2])
        u_t = torch.stack(u_t_tokens, dim=1)
        u_x = torch.stack(u_x_tokens, dim=1)
        u_xx_tokens = []
        for output_index, token_tensor in enumerate(token_indices):
            token = int(token_tensor)
            u_xx_tokens.append(
                torch.autograd.grad(
                    u_x[:, output_index].sum(),
                    sequence,
                    retain_graph=True,
                    create_graph=True,
                )[0][:, token, 1:2]
            )
        u_xx = torch.stack(u_xx_tokens, dim=1)
        return u_t - self.diffusivity * u_xx

    def alignment_loss(self, tx: torch.Tensor) -> torch.Tensor:
        if not isinstance(self.net, PINNMambaExpert):
            return torch.zeros((), device=tx.device, dtype=tx.dtype)
        first = self.net.forward_sequence(tx)
        shifted = tx.clone()
        shifted[:, 0] = shifted[:, 0] + self.net.sequence_step
        second = self.net.forward_sequence(shifted)
        return (first[:, 1:] - second[:, :-1]).square().mean()

    def training_loss(self, data: tuple[torch.Tensor, ...]) -> torch.Tensor:
        initial_tx, initial_u, boundary_tx, boundary_u, collocation = data
        initial = (self(initial_tx) - initial_u).square().mean()
        boundary = (self(boundary_tx) - boundary_u).square().mean()
        physics = self.sequence_residual(collocation).square().mean()
        return initial + boundary + physics + 1000.0 * self.alignment_loss(collocation)


class SingleComplexVectorField(nn.Module):
    """Unitary NDE baseline (currently MLP, the supported complex NDE expert)."""

    def __init__(self, state_dim: int = 2, hidden_dims: tuple[int, ...] = (128, 128, 128)) -> None:
        super().__init__()
        self.net = MLPExpert(state_dim + 1, state_dim, hidden_dims)

    def forward(self, t: torch.Tensor | float, state: torch.Tensor) -> torch.Tensor:
        squeeze = state.ndim == 1
        if squeeze:
            state = state.unsqueeze(0)
        t = torch.as_tensor(t, dtype=state.dtype, device=state.device)
        if t.ndim == 0:
            t = t.expand(len(state), 1)
        elif t.ndim == 1:
            t = t.unsqueeze(-1)
        output = self.net(torch.cat((t, state), dim=-1))
        return output.squeeze(0) if squeeze else output


class SingleMLPHeatPINN(SingleComplexHeatPINN):
    """Backward-compatible name for the former MLP-only baseline."""

    def __init__(
        self,
        diffusivity: float = 0.05,
        hidden_dims: tuple[int, ...] = (128, 128, 128),
    ) -> None:
        super().__init__(
            complex_kind="mlp",
            diffusivity=diffusivity,
            complex_kwargs={"hidden_dims": hidden_dims},
        )


class SingleMLPVectorField(SingleComplexVectorField):
    """Backward-compatible name for the former MLP-only NDE baseline."""
