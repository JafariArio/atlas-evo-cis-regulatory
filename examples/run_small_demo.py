from atlas_evo.sequence_features import basic_sequence_features
from atlas_evo.metrics import regression_metrics

seq = "ACGTACGTACGT"
print(basic_sequence_features(seq))
print(regression_metrics([0.1, 0.6], [0.15, 0.55]))
