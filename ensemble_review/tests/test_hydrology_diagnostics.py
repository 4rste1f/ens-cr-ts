from __future__ import annotations

import unittest

import torch

from complexity_ensemble.hydrology_diagnostics import RoutingSamples, routing_diagnostic


class HydrologyDiagnosticTests(unittest.TestCase):
    def test_detects_aligned_score_and_complex_advantage(self) -> None:
        score = torch.arange(10, dtype=torch.float32)
        samples = RoutingSamples(
            tuple(f"2000-01-{index + 1:02d}" for index in range(10)),
            score,
            torch.sigmoid(score),
            score,
            score + 1.0,
            score,
            score,
            score.clone(),
            score.clone(),
            score.clone(),
            (score >= 8).float(),
        )
        diagnostic = routing_diagnostic(samples, split="test", top_fraction=0.2)
        self.assertAlmostEqual(diagnostic.spearman_complex_advantage, 1.0)
        self.assertAlmostEqual(diagnostic.top_advantage_precision, 1.0)
        self.assertAlmostEqual(diagnostic.complex_better_high_score, 1.0)

    def test_rejects_invalid_fraction(self) -> None:
        values = torch.arange(3, dtype=torch.float32)
        samples = RoutingSamples(("a", "b", "c"), *(values for _ in range(10)))
        with self.assertRaises(ValueError):
            routing_diagnostic(samples, split="test", top_fraction=1.0)


if __name__ == "__main__":
    unittest.main()
