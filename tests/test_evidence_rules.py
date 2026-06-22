from atlas_evo.evidence_audit import classify_evidence


def test_prediction_derived_rule():
    assert classify_evidence({"prediction_derived": True}) == "prediction_derived_capacity"


def test_related_rule():
    assert classify_evidence({"duplicate_or_related": True}) == "related_proxy"
