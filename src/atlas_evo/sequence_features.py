from collections import Counter

_VALID = set("ACGTN")


def clean_sequence(seq):
    seq = str(seq).upper().replace("U", "T")
    return "".join(base if base in _VALID else "N" for base in seq)


def gc_content(seq):
    seq = clean_sequence(seq)
    denom = len(seq)
    if denom == 0:
        return 0.0
    return (seq.count("G") + seq.count("C")) / denom


def kmer_counts(seq, k=6):
    seq = clean_sequence(seq)
    if k <= 0:
        raise ValueError("k must be positive")
    if len(seq) < k:
        return Counter()
    return Counter(seq[i:i+k] for i in range(len(seq)-k+1))


def basic_sequence_features(seq, k=6):
    seq = clean_sequence(seq)
    return {"length": len(seq), "gc_content": gc_content(seq), "n_kmers": max(len(seq)-k+1, 0)}
