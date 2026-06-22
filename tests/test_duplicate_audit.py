import pandas as pd
from atlas_evo.duplicate_leakage_audit import pairwise_overlap


def test_pairwise_overlap():
    a = pd.DataFrame({"sequence": ["AAAA", "CCCC"]})
    b = pd.DataFrame({"sequence": ["AAAA", "GGGG"]})
    out = pairwise_overlap(a, b)
    assert out["seq_overlap_count"] == 1
