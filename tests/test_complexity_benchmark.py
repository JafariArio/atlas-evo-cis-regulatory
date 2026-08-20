from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "12_benchmark_complexity.py"
SPEC = importlib.util.spec_from_file_location("atlas_complexity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class ComplexityBenchmarkTests(unittest.TestCase):
    def test_percentile(self):
        self.assertEqual(MODULE.percentile([1, 2, 3], 0.5), 2)

    def test_loglog_slope(self):
        slope, fit_r2 = MODULE.linear_slope_loglog([(100, 1), (200, 2), (400, 4)])
        self.assertTrue(math.isclose(slope, 1.0, rel_tol=1e-10))
        self.assertTrue(math.isclose(fit_r2, 1.0, rel_tol=1e-10))

    def test_theoretical_components(self):
        components = {row["component"] for row in MODULE.theoretical_complexity_rows()}
        self.assertIn("6-mer counting", components)
        self.assertIn("Combined grammar/proxy MLP head", components)

    def test_threadpool_metadata_is_sanitized(self):
        records = [{"user_api": "blas", "num_threads": 8, "filepath": "private/path"}]
        cleaned = MODULE.sanitize_threadpool_info(records)
        self.assertEqual(cleaned[0]["num_threads"], 8)
        self.assertNotIn("filepath", cleaned[0])

    def test_json_safe_replaces_nonfinite_values(self):
        value = {"peak": float("nan"), "nested": [1.0, float("inf")]}
        expected = {"peak": None, "nested": [1.0, None]}
        self.assertEqual(MODULE.json_safe(value), expected)


if __name__ == "__main__":
    unittest.main()
