import math
from atlas_evo.metrics import regression_metrics


def test_regression_metrics_perfect():
    out = regression_metrics([1, 2, 3], [1, 2, 3])
    assert math.isclose(out["r2"], 1.0)
    assert math.isclose(out["rmse"], 0.0)
    assert math.isclose(out["mae"], 0.0)
