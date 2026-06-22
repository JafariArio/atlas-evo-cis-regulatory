def classify_evidence(row):
    if bool(row.get("prediction_derived", False)):
        return "prediction_derived_capacity"
    if bool(row.get("small_eval", False)):
        return "excluded_sensitivity"
    if bool(row.get("duplicate_or_related", False)):
        return "related_proxy"
    if bool(row.get("measured_target", True)) and bool(row.get("adequate_eval_size", True)):
        return "independent_context"
    return "boundary_or_stress"


def is_transfer_facing(evidence_tier):
    return str(evidence_tier) == "independent_context"
