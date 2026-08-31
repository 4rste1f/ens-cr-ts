from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .routing import MorseRouter


@dataclass
class RoutedOutput:
    value: torch.Tensor
    complex_weight: torch.Tensor
    simple_value: torch.Tensor
    complex_value: torch.Tensor


class HeterogeneousEnsemble(nn.Module):
    """A simple/complex two-expert ensemble with a replaceable router."""

    def __init__(self, simple_expert: nn.Module, complex_expert: nn.Module, router: MorseRouter) -> None:
        super().__init__()
        self.simple_expert = simple_expert
        self.complex_expert = complex_expert
        self.router = router

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        hard: bool = False,
        return_details: bool = False,
    ) -> torch.Tensor | RoutedOutput:
        weight = self.router.complex_weight(inputs, hard=hard)
        simple = self.simple_expert(inputs)
        complex_ = self.complex_expert(inputs)
        value = torch.lerp(simple, complex_, weight.unsqueeze(-1))
        if return_details:
            return RoutedOutput(value, weight, simple, complex_)
        return value

    def interface_loss(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encourage agreement only in the soft transition band."""
        routed = self(inputs, return_details=True)
        assert isinstance(routed, RoutedOutput)
        boundary_weight = 4.0 * routed.complex_weight * (1.0 - routed.complex_weight)
        disagreement = (routed.simple_value - routed.complex_value).square().mean(dim=-1)
        return (boundary_weight * disagreement).mean()

