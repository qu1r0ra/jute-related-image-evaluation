"""Stratified edge sampling for the grouping audit.

Pairs form within a pool. A pair is merged when its Hamming distance is at most the
grouping version's threshold and a near miss up to distance 16. Every pair falls in
one stratum ``pool:band:distance_bin:relation:family``. The design fixes each pool's
budget and allocation, and census strata are taken whole. The sample records every
stratum's population and every pair's inclusion probability, so rates can be
design-weighted afterwards.
"""

import random
from collections import defaultdict
from itertools import combinations, count
from typing import Any

from jute_disease.data.grouping import (
    GROUPING_THRESHOLDS,
    dhash_candidates,
    union_find_components,
)

CONFIRMATORY_POOL = "C"
BANDS = ("merged", "near_miss")
MERGED_BINS = ((0, 2), (3, 4), (5, 8))
NEAR_MISS_BINS = ((5, 8), (9, 12), (13, 16))
ALLOCATIONS = ("proportional", "neyman")
CENSUS_FIELDS = ("pool", "band", "bin", "relation", "family")
DEFAULT_DESIGN: dict[str, Any] = {
    "pools": {
        "A": {"budget": 40, "allocation": "proportional"},
        "B": {"budget": 40, "allocation": "proportional"},
        "C": {"budget": 160, "allocation": "neyman"},
    },
    "census": [{"pool": "C", "band": "merged", "relation": "cross_label"}],
    "neyman_prior": {"merged": 0.05, "near_miss": 0.15},
}


def grouping_threshold(grouping_version: str) -> int:
    """Return the Hamming merge threshold that a grouping version fixes."""
    try:
        return GROUPING_THRESHOLDS[grouping_version]
    except KeyError:
        raise ValueError(
            f"unknown grouping version {grouping_version!r}; "
            f"expected one of {sorted(GROUPING_THRESHOLDS)}"
        ) from None


def merged_bins(threshold: int) -> tuple[tuple[int, int], ...]:
    """Return the merged-band distance bins, clipped to the merge threshold."""
    return tuple(
        (low, min(high, threshold)) for low, high in MERGED_BINS if low <= threshold
    )


def near_miss_bins(threshold: int) -> tuple[tuple[int, int], ...]:
    """Return the near-miss distance bins that lie above the merge threshold."""
    return tuple(
        (max(low, threshold + 1), high)
        for low, high in NEAR_MISS_BINS
        if high > threshold
    )


def validate_design(design: dict[str, Any]) -> dict[str, Any]:
    """Check a stratum design and return it in recorded form."""
    pools = design.get("pools")
    if not isinstance(pools, dict) or not pools:
        raise ValueError("design needs at least one pool")
    for pool, spec in pools.items():
        budget = spec.get("budget") if isinstance(spec, dict) else None
        if not isinstance(budget, int) or budget < 1:
            raise ValueError(f"pool {pool} needs a positive integer budget")
        if spec.get("allocation") not in ALLOCATIONS:
            raise ValueError(f"pool {pool} allocation must be one of {ALLOCATIONS}")
    census = design.get("census", [])
    for selector in census:
        if not selector or not set(selector) <= set(CENSUS_FIELDS):
            raise ValueError(f"census selectors use only {CENSUS_FIELDS}")
    prior = design.get("neyman_prior", {})
    needs_prior = any(spec["allocation"] == "neyman" for spec in pools.values())
    if needs_prior and set(prior) != set(BANDS):
        raise ValueError("Neyman allocation needs a prior rate for each band")
    if any(not 0 < rate < 1 for rate in prior.values()):
        raise ValueError("Neyman prior rates must lie strictly between 0 and 1")
    return {"pools": pools, "census": list(census), "neyman_prior": dict(prior)}


def _allocate(
    populations: dict[str, int], weights: dict[str, float], budget: int
) -> dict[str, int]:
    """Split ``budget`` over strata in proportion to weights.

    Every stratum gets at least one pair and at most its population. Rounding
    follows largest remainders, and a capped stratum's excess flows to the rest.
    """
    allocation = dict.fromkeys(populations, 1)
    remaining = budget - len(populations)
    while remaining > 0:
        open_strata = [s for s in populations if allocation[s] < populations[s]]
        if not open_strata:
            break
        total = sum(weights[s] for s in open_strata)
        raw = {s: remaining * weights[s] / total for s in open_strata}
        give = {
            s: min(int(raw[s]), populations[s] - allocation[s]) for s in open_strata
        }
        leftover = remaining - sum(give.values())
        for stratum in sorted(open_strata, key=lambda s: (int(raw[s]) - raw[s], s)):
            if leftover == 0:
                break
            if allocation[stratum] + give[stratum] < populations[stratum]:
                give[stratum] += 1
                leftover -= 1
        for stratum, amount in give.items():
            allocation[stratum] += amount
        remaining -= sum(give.values())
    return allocation


def _families(
    images: dict[str, tuple[int, str, str]],
    by_pool: dict[str, list[str]],
    threshold: int,
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Group images into families at the threshold and count mixed-label scenes.

    No family is dropped: a family with more than one label is only counted.
    """
    family_of: dict[str, int] = {}
    quality: dict[str, dict[str, int]] = {}
    counter = count()
    for pool, paths in sorted(by_pool.items()):
        hashes = {path: images[path][0] for path in paths}
        components = union_find_components(hashes, dhash_candidates(hashes, threshold))
        for component in components:
            family = next(counter)
            for path in component:
                family_of[path] = family
        quality[pool] = {
            "images": len(paths),
            "families": len(components),
            "mixed_label_scenes": sum(
                len({images[path][1] for path in component}) > 1
                for component in components
            ),
            "dropped_families": 0,
        }
    return family_of, quality


def _census(selectors: list[dict[str, str]], fields: dict[str, str]) -> bool:
    return any(
        all(fields[name] == value for name, value in selector.items())
        for selector in selectors
    )


def sample_edges(
    images: dict[str, tuple[int, str, str]],
    design: dict[str, Any],
    grouping_version: str,
    seed: int = 42,
    inventory_digests: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Draw the audit pairs from a recorded stratum design.

    Each value is ``(hash, label, pool)``. The grouping version sets the threshold.
    Census strata are taken whole; each pool's remaining budget is allocated over
    its other strata proportionally or by Neyman allocation, with at least one pair
    per stratum. Strata with population N and n sampled give each pair an inclusion
    probability n / N.

    ``inventory_digests`` maps each pool to the inventory digest of the freeze its
    images came from. It is recorded so the gate can be bound to that freeze, and
    must name exactly the pools present.
    """
    threshold = grouping_threshold(grouping_version)
    design = validate_design(design)
    if any(len(value) != 3 for value in images.values()):
        raise ValueError("image values must be (hash, label, pool)")
    if any(value[0] < 0 or value[0] >= 1 << 64 for value in images.values()):
        raise ValueError("hashes must be unsigned 64-bit integers")
    by_pool: dict[str, list[str]] = defaultdict(list)
    for path in sorted(images):
        by_pool[str(images[path][2])].append(path)
    if unknown := sorted(set(by_pool) - set(design["pools"])):
        raise ValueError(f"design has no budget for pool(s) {', '.join(unknown)}")
    if inventory_digests is not None and set(inventory_digests) != set(by_pool):
        raise ValueError("inventory digests must name exactly the pools sampled")
    family_of, quality = _families(images, by_pool, threshold)
    bins = {"merged": merged_bins(threshold), "near_miss": near_miss_bins(threshold)}
    rng = random.Random(seed)
    meta: dict[str, dict[str, str]] = {}
    population: dict[str, int] = defaultdict(int)
    is_census: dict[str, bool] = {}
    reservoirs: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for pool in sorted(by_pool):
        capacity = design["pools"][pool]["budget"]
        for a, b in combinations(by_pool[pool], 2):
            distance = (images[a][0] ^ images[b][0]).bit_count()
            band = "merged" if distance <= threshold else "near_miss"
            if band == "near_miss" and distance > 16:
                continue
            low, high = next(
                (low, high) for low, high in bins[band] if low <= distance <= high
            )
            fields = {
                "pool": pool,
                "band": band,
                "bin": f"{low}-{high}",
                "relation": "same_label"
                if images[a][1] == images[b][1]
                else "cross_label",
                "family": "same_family" if family_of[a] == family_of[b] else "split",
            }
            key = ":".join(fields.values())
            if key not in meta:
                meta[key] = fields
                is_census[key] = _census(design["census"], fields)
            population[key] += 1
            bucket = reservoirs[key]
            row = (a, b, distance)
            if is_census[key] or len(bucket) < capacity:
                bucket.append(row)
            else:
                index = rng.randrange(population[key])
                if index < capacity:
                    bucket[index] = row
    strata: dict[str, dict[str, Any]] = {}
    pairs: list[dict[str, Any]] = []
    pools: dict[str, dict[str, Any]] = {}
    for pool in sorted(by_pool):
        spec = design["pools"][pool]
        keys = sorted(k for k, fields in meta.items() if fields["pool"] == pool)
        census_keys = [k for k in keys if is_census[k]]
        rest = [k for k in keys if not is_census[k]]
        census_total = sum(population[k] for k in census_keys)
        weights = {
            k: population[k]
            * (
                (
                    design["neyman_prior"][meta[k]["band"]]
                    * (1 - design["neyman_prior"][meta[k]["band"]])
                )
                ** 0.5
                if spec["allocation"] == "neyman"
                else 1
            )
            for k in rest
        }
        allocation = _allocate(
            {k: population[k] for k in rest},
            weights,
            max(0, spec["budget"] - census_total),
        )
        allocation.update({k: population[k] for k in census_keys})
        for key in keys:
            taken = (
                reservoirs[key]
                if is_census[key]
                else rng.sample(reservoirs[key], allocation[key])
            )
            probability = allocation[key] / population[key]
            strata[key] = {
                **meta[key],
                "population": population[key],
                "sampled": allocation[key],
                "inclusion_probability": probability,
                "census": is_census[key],
            }
            pairs.extend(
                {
                    "image_a": a,
                    "image_b": b,
                    "distance": distance,
                    "stratum": key,
                    "pool": pool,
                    "band": meta[key]["band"],
                    "family": meta[key]["family"],
                    "inclusion_probability": probability,
                }
                for a, b, distance in taken
            )
        pools[pool] = {
            "budget": spec["budget"],
            "allocation": spec["allocation"],
            "census_pairs": census_total,
            "sampled": sum(allocation.values()),
        }
    return {
        "seed": seed,
        "grouping_version": grouping_version,
        "threshold": threshold,
        **({"inventory_digests": dict(inventory_digests)} if inventory_digests else {}),
        "design": design,
        "pools": pools,
        "strata": strata,
        "pairs": pairs,
        "dataset_quality": {
            "pool_c_mixed_label_scenes": quality.get(CONFIRMATORY_POOL, {}).get(
                "mixed_label_scenes", 0
            ),
            "pools": quality,
        },
    }
