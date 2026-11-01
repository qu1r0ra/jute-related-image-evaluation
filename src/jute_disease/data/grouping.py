"""Versioned grouping under the frozen SHA-256 and dHash rule."""

from collections.abc import Iterable
from itertools import combinations

from PIL import Image, ImageOps


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


def flip_canonical_dhash(image: Image.Image) -> int:
    """Return the minimum dHash of the image and its horizontal flip.

    Each dHash converts to grayscale, resizes to 9 by 8 with Lanczos, then
    packs 64 row-major bits (most significant first), one when the left pixel
    is greater than its right neighbor.
    """

    def digest(value: Image.Image) -> int:
        pixels = list(
            value.convert("L")
            .resize((9, 8), Image.Resampling.LANCZOS)
            .get_flattened_data()
        )
        result = 0
        for row in range(8):
            for col in range(8):
                result = (result << 1) | (
                    pixels[row * 9 + col] > pixels[row * 9 + col + 1]
                )
        return result

    return min(digest(image), digest(ImageOps.mirror(image)))


def union_find_components(
    keys: Iterable[str], edges: list[tuple[str, str, int]]
) -> list[tuple[str, ...]]:
    """Return deterministic connected components, including isolated keys."""
    parent = {key: key for key in keys}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for a, b, _ in edges:
        x, y = find(a), find(b)
        parent[max(x, y)] = min(x, y)
    groups: dict[str, list[str]] = {}
    for key in sorted(parent):
        groups.setdefault(find(key), []).append(key)
    return sorted(tuple(group) for group in groups.values())


def audit_components(
    hashes: dict[str, int], labels: dict[str, str], edges: list[tuple[str, str, int]]
) -> list[dict]:
    """Audit component size, diameter and cross-label edges before exclusions.

    Size uses the nearest-rank 99th percentile of component sizes.
    """
    components = union_find_components(hashes, edges)
    if not components:
        return []
    sizes = sorted(len(c) for c in components)
    cutoff = sizes[(99 * len(sizes) + 99) // 100 - 1]
    result = []
    for component in components:
        members = set(component)
        diameter = max(
            (
                (hashes[a] ^ hashes[b]).bit_count()
                for a, b in combinations(component, 2)
            ),
            default=0,
        )
        bridges = sorted(
            (min(a, b), max(a, b))
            for a, b, _ in edges
            if a in members and b in members and labels[a] != labels[b]
        )
        flags = []
        if len(component) > cutoff:
            flags.append("size")
        if diameter > 16:
            flags.append("distance")
        if bridges:
            flags.append("cross_label")
        result.append(
            {
                "members": component,
                "size": len(component),
                "size_percentile_99": cutoff,
                "max_distance": diameter,
                "cross_label_bridges": bridges,
                "flags": flags,
            }
        )
    return result


def repair_components(
    hashes: dict[str, int],
    labels: dict[str, str],
    edges: list[tuple[str, str, int]],
    no_merge: set[tuple[str, str]],
) -> dict:
    """Remove rated edges in distance and pair-hash order, reauditing each step."""
    import hashlib
    import json

    normalized = sorted((min(a, b), max(a, b), d) for a, b, d in edges)
    ratings = {tuple(sorted(pair)) for pair in no_merge}
    removed = []
    history = [audit_components(hashes, labels, normalized)]

    def order(edge: tuple[str, str, int]) -> tuple[int, str]:
        a, b, distance = edge
        pair_hash = hashlib.sha256(
            json.dumps([a, b], separators=(",", ":")).encode()
        ).hexdigest()
        return -distance, pair_hash

    for edge in sorted((edge for edge in normalized if edge[:2] in ratings), key=order):
        normalized.remove(edge)
        removed.append(edge)
        history.append(audit_components(hashes, labels, normalized))
    return {
        "edges": normalized,
        "removed": removed,
        "audit": history[-1],
        "audit_history": history,
    }


def screen_source_overlap(
    source: dict[str, tuple[str, int]], targets: dict[str, dict[str, tuple[str, int]]]
) -> dict[str, list[tuple[str, str, str]]]:
    """Return source exclusions and matching target pools under the frozen rule."""
    result: dict[str, list[tuple[str, str, str]]] = {}
    for source_id, (sha, value) in sorted(source.items()):
        for pool, inventory in sorted(targets.items()):
            for target_id, (target_sha, target_value) in sorted(inventory.items()):
                reason = (
                    "sha256"
                    if sha == target_sha
                    else "dhash"
                    if (value ^ target_value).bit_count() <= 4
                    else None
                )
                if reason:
                    result.setdefault(source_id, []).append((pool, target_id, reason))
    return result


def group_inventory(
    images: dict[str, tuple[str, int, str]],
    no_merge: set[tuple[str, str]] | None = None,
) -> dict:
    """Group inventory paths, audit and repair, then exclude conflicting families."""
    hashes = {key: value for key, (_, value, _) in images.items()}
    labels = {key: label for key, (_, _, label) in images.items()}
    edges = {(a, b): d for a, b, d in dhash_candidates(hashes)}
    exact: dict[str, list[str]] = {}
    for key, (sha, _, _) in sorted(images.items()):
        exact.setdefault(sha, []).append(key)
    for copies in exact.values():
        for a, b in combinations(copies, 2):
            edges[(a, b)] = (hashes[a] ^ hashes[b]).bit_count()
    repair = repair_components(
        hashes,
        labels,
        [(a, b, d) for (a, b), d in edges.items()],
        {(a, b) for a, b in (no_merge or set()) if images[a][0] != images[b][0]},
    )
    excluded = sorted(
        key
        for audit in repair["audit"]
        if len({labels[key] for key in audit["members"]}) > 1
        for key in audit["members"]
    )
    families = [
        audit["members"]
        for audit in repair["audit"]
        if not set(audit["members"]) & set(excluded)
    ]
    return {**repair, "excluded": excluded, "families": families}
