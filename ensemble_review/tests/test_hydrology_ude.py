from __future__ import annotations

import unittest
from datetime import date, timedelta

import torch

from complexity_ensemble.experts import MLPExpert
from complexity_ensemble.hydrology_data import HydrologySeries
from complexity_ensemble.hydrology_ude import (
    RoutedHydrologyUDE,
    SingleComplexHydrologyUDE,
    make_hydrology_ude_data,
)
from complexity_ensemble.hydrology_ude_comparison import train_hydrology_ude


def synthetic_data(days: int = 80):
    dates = tuple(date(2000, 1, 1) + timedelta(days=index) for index in range(days))
    precipitation = torch.tensor([2.0 if index % 5 == 0 else 0.2 for index in range(days)])
    temperature = torch.linspace(-2.0, 12.0, days)
    pet = torch.full((days,), 0.5)
    discharge = 0.5 + 0.1 * precipitation
    series = HydrologySeries(
        dates, discharge, torch.stack((precipitation, temperature, pet), dim=-1),
        ("precipitation", "temperature", "pet"), "synthetic", "test",
    )
    return make_hydrology_ude_data(series)


class HydrologyUDETests(unittest.TestCase):
    def test_vector_field_obeys_water_balance(self) -> None:
        data = synthetic_data()
        model = SingleComplexHydrologyUDE(data.forcing_names, data.forcing_mean, data.forcing_scale)
        state = model.initial_state(4)
        forcing = data.forcings[:4]
        derivative, discharge, evapotranspiration, *_ = model.vector_field(state, forcing)
        precipitation = forcing[:, data.forcing_names.index("precipitation")]
        expected = precipitation - evapotranspiration - discharge
        torch.testing.assert_close(derivative.sum(dim=-1), expected, atol=1e-5, rtol=1e-5)

    def test_rollout_is_differentiable_and_mass_balanced(self) -> None:
        data = synthetic_data()
        model = SingleComplexHydrologyUDE(data.forcing_names, data.forcing_mean, data.forcing_scale)
        result = model.rollout(data.forcings[:12])
        self.assertEqual(result.states.shape, (12, 3))
        self.assertLess(float(result.mass_balance_residual.abs().max().detach()), 1e-4)
        result.discharge.mean().backward()
        self.assertIsNotNone(model.complex_expert.net[0].weight.grad)

    def test_routed_models_share_mlp_complex_expert(self) -> None:
        data = synthetic_data()
        for routing in ("morse", "learned"):
            model = RoutedHydrologyUDE(
                data.forcing_names, data.forcing_mean, data.forcing_scale, routing=routing
            )
            model.fit_router(data.forcings[: data.train_end])
            self.assertIsInstance(model.complex_expert, MLPExpert)
            self.assertEqual(model.rollout(data.forcings[:5]).discharge.shape, (5,))

    def test_short_training_smoke(self) -> None:
        data = synthetic_data()
        model = SingleComplexHydrologyUDE(data.forcing_names, data.forcing_mean, data.forcing_scale)
        train_hydrology_ude(
            model, data, epochs=1, chunk_days=10, warmup_days=5
        )


if __name__ == "__main__":
    unittest.main()
