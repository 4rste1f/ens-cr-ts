from __future__ import annotations

import unittest
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from complexity_ensemble.hydrology_data import HydrologySeries, make_hydrology_data
from examples.compare_learned_morse_hydrology_robustness import (
    EvaluationPeriod,
    coherently_corrupt_split,
    compare_hydrology_models_dose_response,
    parse_period,
    plot_dose_response,
)


def _data(days: int = 90):
    dates = tuple(date(2000, 1, 1) + timedelta(days=index) for index in range(days))
    index = torch.arange(days, dtype=torch.float32)
    series = HydrologySeries(
        dates=dates,
        discharge=1.0 + 0.01 * index,
        forcings=torch.stack((0.2 + index.remainder(5), 5.0 + 0.1 * index, 0.5 + 0.0 * index), dim=1),
        forcing_names=("precipitation", "temperature", "pet"),
        basin_id="test",
        country="Test",
    )
    return make_hydrology_data(series, sequence_length=7, routing_context_length=10)


class CoherentHydrologyRobustnessTests(unittest.TestCase):
    def test_overlapping_windows_share_the_same_noisy_observation(self) -> None:
        data = _data()
        noisy = coherently_corrupt_split(data.train, data, 0.2, seed=123)
        # The final six observations of window zero are the first six of window one.
        torch.testing.assert_close(noisy.inputs[0, 1:], noisy.inputs[1, :-1])

    def test_router_and_expert_views_share_noise(self) -> None:
        data = _data()
        noisy = coherently_corrupt_split(data.train, data, 0.2, seed=123)
        assert noisy.routing_inputs is not None
        torch.testing.assert_close(noisy.inputs, noisy.routing_inputs[:, -7:])

    def test_severity_scales_one_paired_realization(self) -> None:
        data = _data()
        low = coherently_corrupt_split(data.train, data, 0.1, seed=123)
        high = coherently_corrupt_split(data.train, data, 0.2, seed=123)
        # Temperature is unclipped, so doubling severity exactly doubles its perturbation.
        feature = data.feature_names.index("temperature")
        torch.testing.assert_close(
            high.inputs[:, :, feature] - data.train.inputs[:, :, feature],
            2.0 * (low.inputs[:, :, feature] - data.train.inputs[:, :, feature]),
        )

    def test_dose_response_builds_requested_grid(self) -> None:
        data = _data()
        records = compare_hydrology_models_dose_response(
            data,
            period=EvaluationPeriod("all", "2000-01-01", "2000-03-30"),
            models=("morse",), simple_kind="rbf", complex_kind="mlp",
            seeds=(0,), training_noise_levels=(0.0, 0.1),
            inference_noise_levels=(0.0, 0.1, 0.2), inference_noise_draws=2,
            epochs=1, batch_size=128, learning_rate=1e-3,
            consolidation_weight=0.05, device=torch.device("cpu"),
            learned_morse_potential=None,
        )
        self.assertEqual(len(records), 12)
        self.assertEqual(len({record.inference_noise_seed for record in records}), 2)
        with TemporaryDirectory() as directory:
            figure = Path(directory) / "dose_response.png"
            plot_dose_response(records, figure)
            self.assertTrue(figure.exists())

    def test_period_parser(self) -> None:
        self.assertEqual(
            parse_period("wet:2000-01-01:2001-01-01"),
            EvaluationPeriod("wet", "2000-01-01", "2001-01-01"),
        )


if __name__ == "__main__":
    unittest.main()
