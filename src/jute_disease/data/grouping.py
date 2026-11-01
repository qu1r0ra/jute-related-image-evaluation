"""Versioned grouping under the frozen SHA-256 and dHash rule."""

from itertools import combinations


def dhash_candidates(
    hashes: dict[str, int], threshold: int = 4
) -> list[tuple[str, str, int]]:
    """Find all pairs at Hamming distance at most threshold using complete bands."""
    if not 0 <= threshold < 64:
        raise ValueError("threshold must be between 0 and 63")
    if any(value < 0 or value >= 1 << 64 for value in hashes.values()):
        raise ValueError("hashes must be unsigned 64-bit integers")
    bands = threshold + 1
    buckets: dict[tuple[int, int], list[str]] = {}
    for key, value in sorted(hashes.items()):
        for band in range(bands):
            start, end = 64 * band // bands, 64 * (band + 1) // bands
            part = (value >> start) & ((1 << (end - start)) - 1)
            buckets.setdefault((band, part), []).append(key)
    pairs = {pair for bucket in buckets.values() for pair in combinations(bucket, 2)}
    return [
        (a, b, (hashes[a] ^ hashes[b]).bit_count())
        for a, b in sorted(pairs)
        if (hashes[a] ^ hashes[b]).bit_count() <= threshold
    ]
