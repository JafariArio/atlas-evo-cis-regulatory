from atlas_evo.sequence_features import clean_sequence, gc_content, kmer_counts


def test_clean_sequence():
    assert clean_sequence("acgu-x") == "ACGTNN"


def test_gc_content():
    assert gc_content("GGCCAA") == 4 / 6


def test_kmer_counts():
    counts = kmer_counts("AAAA", k=2)
    assert counts["AA"] == 3
