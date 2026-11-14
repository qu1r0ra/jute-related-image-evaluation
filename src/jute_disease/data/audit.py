"""Grouping audit cohorts, blind rating records and their evaluation.

Edge sampling lives in ``audit_sampling``. All samples are deterministic for a
given seed. Ratings use exact and near-duplicate as merge; very similar and
different are no-merge. Every cohort names its grouping version, which fixes the
Hamming threshold, and the agent rater.
"""

import csv
import hashlib
import io
import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from jute_disease.data.audit_sampling import (
    DEFAULT_DESIGN,
    grouping_threshold,
    sample_edges,
)
from jute_disease.data.grouping import flip_canonical_dhash, union_find_components
from jute_disease.evaluation import (
    Interval,
    StratumCounts,
    agreement_gate,
    clopper_pearson_interval,
    design_weighted_interval,
)

RUBRIC = {"exact": 1, "near-duplicate": 1, "very similar": 0, "different": 0}
KAPPA_THRESHOLD = 0.6


def feature_neighbors(
    paths: Sequence[str], extractor: Callable[[Sequence[str]], np.ndarray], k: int = 5
) -> list[dict[str, Any]]:
    """Cosine nearest neighbours from an injected frozen feature extractor."""
    if k < 1 or len(set(paths)) != len(paths):
        raise ValueError("k must be positive and paths unique")
    features = np.asarray(extractor(paths), dtype=float)
    if (
        features.ndim != 2
        or features.shape[0] != len(paths)
        or not np.isfinite(features).all()
    ):
        raise ValueError("extractor must return one finite feature vector per path")
    norms = np.linalg.norm(features, axis=1)
    if np.any(norms == 0):
        raise ValueError("feature vectors must be nonzero")
    features = features / norms[:, None]
    similarity = features @ features.T
    pairs = {}
    for i, path in enumerate(paths):
        for j in sorted(
            (j for j in range(len(paths)) if i != j),
            key=lambda j: (-similarity[i, j], paths[j]),
        )[:k]:
            pair = tuple(sorted((path, paths[j])))
            pairs[pair] = float(similarity[i, j])
    return [
        {"image_a": a, "image_b": b, "cosine": value}
        for (a, b), value in sorted(pairs.items())
    ]


class ImageNetExtractor:
    """Load frozen ImageNet features and expose a digest of the loaded weights."""

    def __init__(self) -> None:
        self.provenance: dict[str, str] = {}

    def __call__(self, paths: Sequence[str]) -> np.ndarray:
        import timm
        import torch

        tag = "mobilenetv2_100.ra_in1k"
        model = timm.create_model(tag, pretrained=True, num_classes=0).eval()
        digest = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(str(tensor.dtype).encode("ascii") + b"\0")
            digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        self.provenance = {
            "kind": "imagenet",
            "model": tag,
            "timm_version": timm.__version__,
            "weights_sha256": digest.hexdigest(),
            "digest_method": "sorted state_dict names, dtype, shape and tensor bytes",
        }
        transform = timm.data.create_transform(
            **timm.data.resolve_model_data_config(model), is_training=False
        )
        vectors = []
        with torch.inference_mode():
            for path in paths:
                with Image.open(path) as image:
                    vectors.append(
                        model(transform(image.convert("RGB")).unsqueeze(0))
                        .squeeze(0)
                        .cpu()
                        .numpy()
                    )
        return np.asarray(vectors)


def residual_sample(
    pools: dict[str, dict[str, int]],
    training: dict[str, dict[str, int]],
    extractor: Callable[[Sequence[str]], np.ndarray],
    k: int = 5,
    count: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Uniform protected-test sample and top-k train neighbours per pool."""
    if k < 1 or count < 1 or set(pools) != set(training):
        raise ValueError("positive k and count and matching pools required")
    for pool in sorted(pools):
        test, train = pools[pool], training[pool]
        if len(test) < count:
            raise ValueError(
                f"pool {pool} has {len(test)} protected-test images; {count} required"
            )
        if len(train) < k:
            raise ValueError(
                f"pool {pool} has {len(train)} training images; "
                f"{k} required for neighbours"
            )
        if set(test) & set(train):
            raise ValueError("test and train must be disjoint")
    rng = random.Random(seed)
    result = {}
    for pool in sorted(pools):
        test, train = pools[pool], training[pool]
        chosen = rng.sample(sorted(test), count)
        paths = chosen + sorted(train)
        vectors = np.asarray(extractor(paths), dtype=float)
        if (
            vectors.ndim != 2
            or len(vectors) != len(paths)
            or not np.isfinite(vectors).all()
            or np.any(np.linalg.norm(vectors, axis=1) == 0)
        ):
            raise ValueError("invalid feature vectors")
        vectors /= np.linalg.norm(vectors, axis=1)[:, None]
        rows = []
        for i, image in enumerate(chosen):
            hash_order = sorted(
                train, key=lambda p: ((test[image] ^ train[p]).bit_count(), p)
            )[:k]
            feature_order = sorted(
                range(len(chosen), len(paths)),
                key=lambda j: (-float(vectors[i] @ vectors[j]), paths[j]),
            )[:k]
            rows.append(
                {
                    "test": image,
                    "hash_neighbors": [
                        {"image": p, "distance": (test[image] ^ train[p]).bit_count()}
                        for p in hash_order
                    ],
                    "feature_neighbors": [
                        {"image": paths[j], "cosine": float(vectors[i] @ vectors[j])}
                        for j in feature_order
                    ],
                }
            )
        result[pool] = {
            "population": len(test),
            "sample_size": len(chosen),
            "rows": rows,
        }
    return {
        "seed": seed,
        "interpretation": "retrieval-conditional lower bound",
        "pools": result,
    }


def synthetic_recall(
    images: Sequence[Image.Image], grouping_version: str
) -> dict[str, Any]:
    """Audit 100 unrelated sources under the prespecified mild transforms.

    The grouping version fixes the Hamming threshold. Crop strength is the
    fraction of total image area targeted for removal. Rounded-up centered
    dimensions ensure the actual loss does not exceed it.
    """
    threshold = grouping_threshold(grouping_version)
    if len(images) != 100:
        raise ValueError(
            "synthetic recall requires exactly 100 unrelated source images"
        )
    hashes = [flip_canonical_dhash(im) for im in images]
    rows = []
    for name, strengths in (
        ("resize", (0.5, 0.75)),
        ("crop", (0.05, 0.10)),
        ("jpeg", (70, 85)),
    ):
        for strength in strengths:
            transformed = []
            removed = []
            for image in images:
                if name == "resize":
                    copy = image.resize(
                        (
                            max(1, round(image.width * strength)),
                            max(1, round(image.height * strength)),
                        ),
                        Image.Resampling.LANCZOS,
                    )
                elif name == "crop":
                    factor = math.sqrt(1 - strength)
                    width, height = (
                        math.ceil(image.width * factor),
                        math.ceil(image.height * factor),
                    )
                    left, top = (image.width - width) // 2, (image.height - height) // 2
                    copy = image.crop((left, top, left + width, top + height))
                    removed.append(1 - width * height / (image.width * image.height))
                else:
                    buffer = io.BytesIO()
                    image.convert("RGB").save(
                        buffer, format="JPEG", quality=int(strength)
                    )
                    buffer.seek(0)
                    with Image.open(buffer) as jpeg:
                        copy = jpeg.convert("RGB")
                transformed.append(copy)
            hits = sum(
                (h ^ flip_canonical_dhash(copy)).bit_count() <= threshold
                for h, copy in zip(hashes, transformed, strict=True)
            )
            row = {
                "transform": name,
                "strength": strength,
                "hits": hits,
                "total": len(images),
                "recall": hits / len(images),
                "mild": True,
            }
            if removed:
                row["actual_removed_area"] = removed
                row["actual_removed_area_min"] = min(removed)
                row["actual_removed_area_max"] = max(removed)
            rows.append(row)
    false_merges = sum(
        (a ^ b).bit_count() <= threshold for a, b in combinations(hashes, 2)
    )
    total = len(images) * (len(images) - 1) // 2
    rate = false_merges / total
    mild_failed = any(row["mild"] and row["recall"] < 0.9 for row in rows)
    false_failed = rate > 0.05
    return {
        "source_images": len(images),
        "transforms": rows,
        "unrelated": {"false_merges": false_merges, "total": total, "rate": rate},
        "threshold_change": {
            "mild_recall_failed": mild_failed,
            "unrelated_false_merge_failed": false_failed,
            "condition_met": mild_failed or false_failed,
        },
    }


def prepare_cohort(
    samples: dict[str, Any], agent: dict[str, str], seed: int = 42
) -> dict[str, Any]:
    """Take every sampled pair into one blind cohort, shuffled and numbered.

    The cohort carries what the evaluation needs: each pair's stratum, pool,
    family and inclusion probability, the stratum populations, the grouping
    version with its derived threshold, and the agent rater. Agent ratings are
    left blank for independent review.
    """
    version = samples.get("grouping_version")
    threshold = grouping_threshold(version)
    if samples.get("threshold") != threshold:
        raise ValueError("edge samples do not match their grouping version's threshold")
    if not all(agent.get(key, "").strip() for key in ("name", "version")):
        raise ValueError("the agent rater needs a name and a version")
    strata = samples["strata"]
    seen: set[tuple[str, str]] = set()
    counts: dict[str, int] = defaultdict(int)
    rows = []
    for pair in samples["pairs"]:
        a, b, distance = pair["image_a"], pair["image_b"], pair["distance"]
        stratum = strata.get(pair["stratum"])
        if a == b or tuple(sorted((a, b))) in seen:
            raise ValueError("edge samples must contain distinct unique pairs")
        if stratum is None or (stratum["band"] == "merged") != (distance <= threshold):
            raise ValueError("edge distance does not match its stratum")
        if distance > 16:
            raise ValueError("edge distance does not match its stratum")
        seen.add(tuple(sorted((a, b))))
        counts[pair["stratum"]] += 1
        rows.append(
            {
                "pair_id": "",
                "image_a": a,
                "image_b": b,
                "distance": distance,
                "agent_rating": "",
                "algorithm_merge": stratum["band"] == "merged",
                "stratum": pair["stratum"],
                "pool": stratum["pool"],
                "band": stratum["band"],
                "family": stratum["family"],
                "inclusion_probability": pair["inclusion_probability"],
            }
        )
    if set(counts) != set(strata):
        raise ValueError("each population stratum must be represented in the cohort")
    denominators = {key: detail["population"] for key, detail in strata.items()}
    _check_inclusion(rows, denominators)
    rng = random.Random(seed)
    rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["pair_id"] = str(index)
    return {
        "pairs": rows,
        "denominators": denominators,
        "grouping_version": version,
        "threshold": threshold,
        "agent": {"name": agent["name"], "version": agent["version"]},
        "dataset_quality": samples["dataset_quality"],
        "selection": {
            "seed": seed,
            "method": "every sampled pair, shuffled",
            "sample_seed": samples["seed"],
            "design": samples["design"],
        },
    }


def _check_inclusion(
    rows: Sequence[dict[str, Any]], denominators: dict[str, int]
) -> None:
    sampled: dict[str, int] = defaultdict(int)
    for row in rows:
        sampled[row["stratum"]] += 1
    for row in rows:
        stratum = row["stratum"]
        if stratum not in denominators or not math.isclose(
            row["inclusion_probability"], sampled[stratum] / denominators[stratum]
        ):
            raise ValueError("inclusion probability does not match the stratum counts")


def export_blind(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """Write only pair identifiers and a blank owner rating."""
    ids = [row["pair_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("pair IDs must be unique")
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["pair_id", "image_a", "image_b", "rating"]
        )
        writer.writeheader()
        writer.writerows(
            {
                "pair_id": row["pair_id"],
                "image_a": row["image_a"],
                "image_b": row["image_b"],
                "rating": "",
            }
            for row in rows
        )


def _clusters(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Label each pair by the component of pairs that share an image.

    Components form within a pool and band, the unit the bootstrap resamples in.
    """
    by_group: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    for row in rows:
        by_group[f"{row['pool']}:{row['band']}"].append(
            (row["image_a"], row["image_b"], 0)
        )
    label: dict[tuple[str, str], str] = {}
    for group, edges in by_group.items():
        keys = {image for a, b, _ in edges for image in (a, b)}
        for index, component in enumerate(union_find_components(keys, edges)):
            for image in component:
                label[group, image] = f"{group}#{index}"
    return [label[f"{row['pool']}:{row['band']}", row["image_a"]] for row in rows]


def _specific_agreement(
    first: Sequence[int], second: Sequence[int]
) -> dict[str, float | None]:
    positive = sum(a == b == 1 for a, b in zip(first, second, strict=True))
    negative = sum(a == b == 0 for a, b in zip(first, second, strict=True))
    discordant = len(first) - positive - negative
    return {
        "positive": 2 * positive / (2 * positive + discordant)
        if positive + discordant
        else None,
        "negative": 2 * negative / (2 * negative + discordant)
        if negative + discordant
        else None,
        "pairs": len(first),
    }


BOUND_METHODS = {
    "per_stratum": "exact two-sided 95% Clopper-Pearson",
    "design_weighted": (
        "conservative two-sided 95%: exact finite-population (hypergeometric) "
        "bounds per sampled stratum at tail 0.025 / m, summed with population "
        "weights (union bound over the m sampled strata); coverage is at least 95%"
    ),
}


def _interval(estimate: Interval) -> dict[str, float]:
    return {
        "rate": estimate.estimate,
        "lower": estimate.lower,
        "upper": estimate.upper,
    }


def _error_rates(
    rows: Sequence[dict[str, Any]],
    owner: Sequence[int],
    denominators: dict[str, int],
) -> dict[str, Any]:
    """Per-stratum exact and conservative design-weighted error rates.

    Covers wrong-merge and false-split rates.

    A wrong merge is a merged pair the owner rates no-merge. A false split is a
    near-miss pair in different families that the owner rates a duplicate.
    """
    events: dict[str, int] = defaultdict(int)
    sampled: dict[str, int] = defaultdict(int)
    kinds: dict[str, str] = {}
    pools: dict[str, str] = {}
    for row, rating in zip(rows, owner, strict=True):
        key = row["stratum"]
        sampled[key] += 1
        pools[key] = row["pool"]
        if row["band"] == "merged":
            kinds[key] = "wrong_merge"
            events[key] += int(rating == 0)
        else:
            kinds[key] = "false_split" if row["family"] == "split" else "same_family"
            events[key] += int(rating == 1)
    per_stratum = {}
    for key in sorted(sampled):
        exact = clopper_pearson_interval(events[key], sampled[key])
        per_stratum[key] = {
            "kind": kinds[key],
            "pool": pools[key],
            "population": denominators[key],
            "sampled": sampled[key],
            "events": events[key],
            **_interval(exact),
        }

    def weighted(kind: str, pool: str | None) -> dict[str, Any] | None:
        chosen = [
            key
            for key in sorted(sampled)
            if kinds[key] == kind and pool in (None, pools[key])
        ]
        if not chosen:
            return None
        counts = [
            StratumCounts(denominators[key], sampled[key], events[key])
            for key in chosen
        ]
        return {
            **_interval(design_weighted_interval(counts)),
            "population": sum(c.population for c in counts),
            "sampled": sum(c.sampled for c in counts),
            "events": sum(c.events for c in counts),
        }

    pool_names = sorted(set(pools.values()))
    return {
        "bound_methods": BOUND_METHODS,
        "per_stratum": per_stratum,
        "wrong_merge": {
            "overall": weighted("wrong_merge", None),
            "by_pool": {p: weighted("wrong_merge", p) for p in pool_names},
        },
        "false_split": {
            "overall": weighted("false_split", None),
            "by_pool": {p: weighted("false_split", p) for p in pool_names},
        },
    }


def cohort_digest(cohort: dict[str, Any]) -> str:
    identity = {
        "pairs": sorted(
            (
                {
                    key: row[key]
                    for key in (
                        "pair_id",
                        "image_a",
                        "image_b",
                        "stratum",
                        "algorithm_merge",
                        "inclusion_probability",
                    )
                }
                for row in cohort["pairs"]
            ),
            key=lambda row: row["pair_id"],
        ),
        "denominators": cohort["denominators"],
        "grouping_version": cohort["grouping_version"],
        "threshold": cohort["threshold"],
        "agent": cohort["agent"],
    }
    return hashlib.sha256(
        json.dumps(
            identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def evaluate_ratings(
    cohort: dict[str, Any],
    owner_path: Path,
    round_number: int = 1,
    seed: int = 42,
    previous: dict[str, Any] | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Import owner ratings for a cohort and report agreement and error rates.

    The cohort must name its grouping version and agent rater. A single retry is
    bound to the same cohort by digest.
    """
    if round_number not in (1, 2):
        raise ValueError("only one rubric revision is permitted")
    for key in ("grouping_version", "agent", "pairs", "denominators"):
        if key not in cohort:
            raise ValueError(f"cohort lacks {key}; rebuild it with audit-cohort")
    if grouping_threshold(cohort["grouping_version"]) != cohort.get("threshold"):
        raise ValueError("cohort threshold does not match its grouping version")
    if not all(str(cohort["agent"].get(k, "")).strip() for k in ("name", "version")):
        raise ValueError("cohort does not name the agent rater")
    rows, denominators = cohort["pairs"], cohort["denominators"]
    _check_inclusion(rows, denominators)
    digest = cohort_digest(cohort)
    if round_number == 2:
        if (
            previous is None
            or previous.get("round") != 1
            or previous.get("pass") is not False
            or not revision
            or not revision.strip()
        ):
            raise ValueError(
                "retry requires a failed first round and a rubric revision"
            )
        if previous.get("cohort_digest") != digest:
            raise ValueError("retry cohort does not match the first round")
    elif previous is not None or revision is not None:
        raise ValueError("previous round and revision apply only to a retry")
    with Path(owner_path).open(newline="", encoding="utf-8") as handle:
        owner = list(csv.DictReader(handle))
    by_id = {row["pair_id"]: row for row in rows}
    if (
        len(by_id) != len(rows)
        or len(owner) != len(rows)
        or len({r["pair_id"] for r in owner}) != len(owner)
        or {r["pair_id"] for r in owner} != set(by_id)
    ):
        raise ValueError("owner ratings must match the sampled pair IDs exactly")
    ordered = []
    first, second = [], []
    similar = []
    for rating in owner:
        row = by_id[rating["pair_id"]]
        if rating["image_a"] != row["image_a"] or rating["image_b"] != row["image_b"]:
            raise ValueError("pair images do not match")
        if rating["rating"] not in RUBRIC or row["agent_rating"] not in RUBRIC:
            raise ValueError("invalid rubric rating")
        ordered.append(row)
        first.append(RUBRIC[row["agent_rating"]])
        second.append(RUBRIC[rating["rating"]])
        if "very similar" in (rating["rating"], row["agent_rating"]):
            similar.append(row["pair_id"])
    if {row["stratum"] for row in ordered} != set(denominators) or any(
        n <= 0 for n in denominators.values()
    ):
        raise ValueError(
            "every population stratum must be represented with positive denominator"
        )
    if len(set(first + second)) == 1:
        kappa, interval, passed = None, None, False
        method = "Cohen kappa undefined: both raters used one category"
    else:
        agreement = agreement_gate(
            first,
            second,
            threshold=KAPPA_THRESHOLD,
            seed=seed,
            strata=[f"{row['pool']}:{row['band']}" for row in ordered],
            clusters=_clusters(ordered),
        )
        kappa = agreement.kappa
        interval = [agreement.interval.lower, agreement.interval.upper]
        passed, method = agreement.passed, agreement.method
    by_pool = {
        pool: _specific_agreement(
            [a for a, row in zip(first, ordered, strict=True) if row["pool"] == pool],
            [b for b, row in zip(second, ordered, strict=True) if row["pool"] == pool],
        )
        for pool in sorted({row["pool"] for row in ordered})
    }
    disagreement = design_weighted_interval(
        [
            StratumCounts(
                denominators[key],
                sum(row["stratum"] == key for row in ordered),
                sum(
                    a != b
                    for a, b, row in zip(first, second, ordered, strict=True)
                    if row["stratum"] == key
                ),
            )
            for key in sorted(denominators)
        ]
    )
    return {
        "grouping_version": cohort["grouping_version"],
        "hamming_threshold": cohort["threshold"],
        "agent": cohort["agent"],
        "dataset_quality": cohort.get("dataset_quality"),
        "cohort_digest": digest,
        "revision": revision,
        "kappa": kappa,
        "kappa_interval": interval,
        "threshold": KAPPA_THRESHOLD,
        "pass": passed,
        "method": method,
        "positive_agreement": _specific_agreement(first, second)["positive"],
        "negative_agreement": _specific_agreement(first, second)["negative"],
        "specific_agreement_by_pool": by_pool,
        **_error_rates(ordered, second, denominators),
        "agent_owner_disagreement": _interval(disagreement),
        "very_similar": similar,
        "round": round_number,
        "next_action": "proceed"
        if passed
        else "revise_rubric_and_rerun"
        if round_number == 1
        else "owner_escalation",
    }


def fixture_round(output: Path, grouping_version: str) -> dict[str, Any]:
    """Create and import a synthetic blind rating round without model weights."""
    grouping_threshold(grouping_version)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    values = random.Random(42).sample(range(1 << 20), 128)
    for i in range(16):
        # Near copies (0 to 3 bits apart) give the merged band its closest bin.
        values[i + 64] = values[i] ^ ((1 << (i % 4)) - 1)
    hashes = {
        f"image-{i}": (value, "ab"[(i + i // 64) % 2], "ABC"[i % 64 % 3])
        for i, value in enumerate(values)
    }

    def extractor(paths: Sequence[str]) -> np.ndarray:
        return np.asarray([[int(path.split("-")[-1]) + 1, 1] for path in paths])

    paths = list(hashes)
    rng = np.random.default_rng(42)
    images = [
        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
        for _ in range(100)
    ]
    samples = {
        "edges": sample_edges(hashes, DEFAULT_DESIGN, grouping_version),
        "features": feature_neighbors(paths, extractor),
        "feature_provenance": {"kind": "synthetic", "version": "fixture-v1"},
        "residual": residual_sample(
            {"fixture": {f"image-{i}": i for i in range(220)}},
            {"fixture": {f"image-{i}": i for i in range(220, 230)}},
            extractor,
        ),
        "recall": synthetic_recall(images, grouping_version),
    }
    (output / "samples.json").write_text(
        json.dumps(samples, indent=2) + "\n", encoding="utf-8"
    )
    cohort = prepare_cohort(
        samples["edges"], {"name": "synthetic-fixture-agent", "version": "fixture-v1"}
    )
    rows = cohort["pairs"]
    for row in rows:
        row["agent_rating"] = (
            "near-duplicate" if row["algorithm_merge"] else "different"
        )
    (output / "agent.json").write_text(
        json.dumps(cohort, indent=2) + "\n", encoding="utf-8"
    )
    export_blind(rows, output / "owner.csv")
    with (output / "owner.csv").open(newline="", encoding="utf-8") as handle:
        owner = list(csv.DictReader(handle))
    for row, original in zip(owner, rows, strict=True):
        row["rating"] = (
            "very similar" if not original["algorithm_merge"] else "near-duplicate"
        )
    with (output / "synthetic_ratings.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["pair_id", "image_a", "image_b", "rating"]
        )
        writer.writeheader()
        writer.writerows(owner)
    for rating, original in zip(owner, rows, strict=True):
        rating["rating"] = "different" if original["algorithm_merge"] else "exact"
    failed_path = output / "synthetic_failed_ratings.csv"
    with failed_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["pair_id", "image_a", "image_b", "rating"]
        )
        writer.writeheader()
        writer.writerows(owner)
    first_failed = evaluate_ratings(cohort, failed_path, round_number=1)
    failed = [
        first_failed,
        evaluate_ratings(
            cohort,
            failed_path,
            round_number=2,
            previous=first_failed,
            revision="synthetic fixture revision",
        ),
    ]
    (output / "failed_rounds.json").write_text(
        json.dumps(failed, indent=2) + "\n", encoding="utf-8"
    )
    report = evaluate_ratings(cohort, output / "synthetic_ratings.csv")
    (output / "agreement.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
