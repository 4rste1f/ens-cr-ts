from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from complexity_ensemble.baselines import SingleComplexHeatPINN
from complexity_ensemble.comparison import ComparisonRecord, comparison_summary, save_comparison
from complexity_ensemble.experts import MLPExpert
from complexity_ensemble.pinn import HeatPINN
from complexity_ensemble.pinnmamba import PINNMambaExpert


class ComplexComparisonTests(unittest.TestCase):
    def test_all_heat_candidates_use_selected_complex_architecture(self) -> None:
        for complex_kind, expert_type in (
            ("mlp", MLPExpert),
            ("pinnmamba", PINNMambaExpert),
        ):
            with self.subTest(complex_kind=complex_kind):
                morse = HeatPINN(complex_kind=complex_kind, routing="morse")
                learned = HeatPINN(complex_kind=complex_kind, routing="learned")
                single = SingleComplexHeatPINN(complex_kind=complex_kind)
                self.assertIsInstance(morse.ensemble.complex_expert, expert_type)
                self.assertIsInstance(learned.ensemble.complex_expert, expert_type)
                self.assertIsInstance(single.net, expert_type)

    def test_reports_single_complex_and_selected_architecture(self) -> None:
        records = [
            ComparisonRecord(
                "heat", model, "none" if model == "single_complex" else "rbf",
                "pinnmamba", 0, 0.0, 0.0, 1.0, 1.0, 1.0, 1, 1.0,
            )
            for model in ("morse", "learned", "single_complex")
        ]
        summary = comparison_summary(records)
        self.assertIn("complex expert: pinnmamba", summary)
        self.assertIn("single_complex", summary)
        self.assertNotIn("single_mlp", summary)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.csv"
            save_comparison(records, output)
            self.assertIn("complex_expert", output.read_text(encoding="utf-8").splitlines()[0])


if __name__ == "__main__":
    unittest.main()
