"""Grouping audit sampling and blind rating records.

All samples are deterministic for a given seed. Ratings use exact and
near-duplicate as merge; very similar and different are no-merge.
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

from jute_disease.data.grouping import flip_canonical_dhash
from jute_disease.evaluation import agreement_gate

RUBRIC = {"exact": 1, "near-duplicate": 1, "very similar": 0, "different": 0}


def sample_edges(
    images: dict[str, tuple[int, str]], count: int = 200, seed: int = 42
) -> dict[str, Any]:
    """Sample each distance band uniformly and record population strata."""
    if count < 1:
        raise ValueError("count must be positive")
    if any(value < 0 or value >= 1 << 64 for value, _ in images.values()):
        raise ValueError("hashes must be unsigned 64-bit integers")
    rng = random.Random(seed)
    reservoirs: dict[str, dict[str, list[dict[str, Any]]]] = {
        "near_miss": defaultdict(list),
        "merged": defaultdict(list),
    }
    populations: dict[str, dict[str, int]] = {
        "near_miss": defaultdict(int),
        "merged": defaultdict(int),
    }
    for a, b in combinations(sorted(images), 2):
        distance = (images[a][0] ^ images[b][0]).bit_count()
        band = (
            "near_miss"
            if 5 <= distance <= 16
            else "merged"
            if 3 <= distance <= 4
            else None
        )
        if band:
            relation = "same_label" if images[a][1] == images[b][1] else "cross_label"
            stratum = f"{distance}:{relation}"
            population = populations[band]
            population[stratum] += 1
            bucket = reservoirs[band][stratum]
            row = {"image_a": a, "image_b": b, "distance": distance, "stratum": stratum}
            if len(bucket) < count:
                bucket.append(row)
            else:
                index = rng.randrange(population[stratum])
                if index < count:
                    bucket[index] = row
    result = {}
    for band in reservoirs:
        population, buckets = populations[band], reservoirs[band]
        target = min(count, sum(population.values()))
        allocation = dict.fromkeys(sorted(population), 0)
        while sum(allocation.values()) < target:
            for stratum in allocation:
                if sum(allocation.values()) == target:
                    break
                if allocation[stratum] < population[stratum]:
                    allocation[stratum] += 1
        sampled = []
        for stratum, amount in allocation.items():
            sampled.extend(rng.sample(buckets[stratum], amount))
        result[band] = {
            "population": sum(population.values()),
            "strata": {
                s: {"population": population[s], "sampled": allocation[s]}
                for s in allocation
            },
            "pairs": sampled,
        }
    return {"seed": seed, "bands": result}


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


def synthetic_recall(images: Sequence[Image.Image]) -> dict[str, Any]:
    """Audit 100 unrelated sources under the prespecified mild transforms.

    Crop strength is the fraction of total image area targeted for removal.
    Rounded-up centered dimensions ensure the actual loss does not exceed it.
    """
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
                (h ^ flip_canonical_dhash(copy)).bit_count() <= 4
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
    false_merges = sum((a ^ b).bit_count() <= 4 for a, b in combinations(hashes, 2))
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


def prepare_cohort(samples: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    """Select 60 pairs per band, round-robin across shuffled strata.

    Agent ratings are left blank for independent review. Every population
    stratum must have a sampled pair so weighted errors remain estimable.
    """
    rng = random.Random(seed)
    rows = []
    denominators = {}
    seen = set()
    for band_name in ("merged", "near_miss"):
        band = samples["bands"][band_name]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for pair in band["pairs"]:
            a, b, distance = pair["image_a"], pair["image_b"], pair["distance"]
            if a == b or tuple(sorted((a, b))) in seen:
                raise ValueError("edge samples must contain distinct unique pairs")
            if not (
                3 <= distance <= 4 if band_name == "merged" else 5 <= distance <= 16
            ):
                raise ValueError("edge distance does not match its band")
            seen.add(tuple(sorted((a, b))))
            grouped[pair["stratum"]].append(pair)
        strata = band["strata"]
        if set(grouped) != set(strata) or len(strata) > 60:
            raise ValueError(
                "each population stratum must be represented in the owner cohort"
            )
        if len(band["pairs"]) < 60:
            raise ValueError(f"{band_name} needs at least 60 sampled pairs")
        for stratum, detail in strata.items():
            if detail["population"] < len(grouped[stratum]) or detail["sampled"] != len(
                grouped[stratum]
            ):
                raise ValueError("stratum counts do not match sampled pairs")
            denominators[f"{band_name}:{stratum}"] = detail["population"]
            rng.shuffle(grouped[stratum])
        selected = []
        while len(selected) < 60:
            for stratum in sorted(grouped):
                if grouped[stratum] and len(selected) < 60:
                    selected.append(grouped[stratum].pop())
        for pair in selected:
            rows.append(
                {
                    "pair_id": str(len(rows)),
                    "image_a": pair["image_a"],
                    "image_b": pair["image_b"],
                    "agent_rating": "",
                    "algorithm_merge": band_name == "merged",
                    "stratum": f"{band_name}:{pair['stratum']}",
                }
            )
    rng.shuffle(rows)
    for index, row in enumerate(rows):
        row["pair_id"] = str(index)
    return {
        "pairs": rows,
        "denominators": denominators,
        "selection": {
            "seed": seed,
            "method": "60 per band; shuffled within strata, round-robin, then shuffled",
            "sample_seed": samples["seed"],
        },
    }


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


def evaluate_ratings(
    rows: Sequence[dict[str, Any]],
    owner_path: Path,
    denominators: dict[str, int],
    round_number: int = 1,
    seed: int = 42,
    previous: dict[str, Any] | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Import ratings; bind a single retry to the same pair cohort and strata."""
    if round_number not in (1, 2):
        raise ValueError("only one rubric revision is permitted")
    identity = {
        "pairs": sorted(
            (
                {key: row[key] for key in ("pair_id", "image_a", "image_b", "stratum")}
                | {"algorithm_merge": row.get("algorithm_merge")}
                for row in rows
            ),
            key=lambda row: row["pair_id"],
        ),
        "denominators": denominators,
    }
    digest = hashlib.sha256(
        json.dumps(
            identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
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
    first, second = [], []
    errors: dict[str, list[int]] = defaultdict(list)
    false_merges: dict[str, list[int]] = defaultdict(list)
    missed_merges: dict[str, list[int]] = defaultdict(list)
    algorithm_errors: dict[str, list[int]] = defaultdict(list)
    algorithm_false: dict[str, list[int]] = defaultdict(list)
    algorithm_missed: dict[str, list[int]] = defaultdict(list)
    has_algorithm = all("algorithm_merge" in row for row in rows)
    if any("algorithm_merge" in row for row in rows) and not has_algorithm:
        raise ValueError("algorithm decisions must be supplied for all pairs")
    similar = []
    for rating in owner:
        row = by_id[rating["pair_id"]]
        if rating["image_a"] != row["image_a"] or rating["image_b"] != row["image_b"]:
            raise ValueError("pair images do not match")
        if rating["rating"] not in RUBRIC or row["agent_rating"] not in RUBRIC:
            raise ValueError("invalid rubric rating")
        a, b = RUBRIC[row["agent_rating"]], RUBRIC[rating["rating"]]
        first.append(a)
        second.append(b)
        errors[row["stratum"]].append(int(a != b))
        false_merges[row["stratum"]].append(int(a == 1 and b == 0))
        missed_merges[row["stratum"]].append(int(a == 0 and b == 1))
        if has_algorithm:
            prediction = row["algorithm_merge"]
            if not isinstance(prediction, bool):
                raise ValueError("algorithm_merge must be boolean")
            algorithm_errors[row["stratum"]].append(int(prediction != b))
            algorithm_false[row["stratum"]].append(int(prediction and not b))
            algorithm_missed[row["stratum"]].append(int(not prediction and b))
        if "very similar" in (rating["rating"], row["agent_rating"]):
            similar.append(row["pair_id"])
    if set(errors) != set(denominators) or any(n <= 0 for n in denominators.values()):
        raise ValueError(
            "every population stratum must be represented with positive denominator"
        )
    positive = sum(a == b == 1 for a, b in zip(first, second, strict=True))
    negative = sum(a == b == 0 for a, b in zip(first, second, strict=True))
    discordant = len(first) - positive - negative
    pos_denom, neg_denom = 2 * positive + discordant, 2 * negative + discordant
    positive_agreement = 2 * positive / pos_denom if pos_denom else None
    negative_agreement = 2 * negative / neg_denom if neg_denom else None
    if len(set(first + second)) == 1:
        kappa, interval, passed = None, None, False
        threshold = 0.6
        method = "Cohen kappa undefined: both raters used one category"
    else:
        agreement = agreement_gate(first, second, seed=seed)
        kappa = agreement.kappa
        interval = [agreement.interval.lower, agreement.interval.upper]
        threshold, passed, method = (
            agreement.threshold,
            agreement.passed,
            agreement.method,
        )
    weighted = sum(denominators[s] * sum(e) / len(e) for s, e in errors.items()) / sum(
        denominators.values()
    )
    total = sum(denominators.values())
    false_rate = (
        sum(denominators[s] * sum(e) / len(e) for s, e in false_merges.items()) / total
    )
    missed_rate = (
        sum(denominators[s] * sum(e) / len(e) for s, e in missed_merges.items()) / total
    )

    def weighted_rate(values: dict[str, list[int]]) -> float:
        return sum(denominators[s] * sum(e) / len(e) for s, e in values.items()) / total

    algorithm = {}
    if has_algorithm:
        algorithm = {
            "weighted_algorithm_error_rate": weighted_rate(algorithm_errors),
            "weighted_false_merge_rate": weighted_rate(algorithm_false),
            "weighted_false_split_rate": weighted_rate(algorithm_missed),
        }
    return {
        **algorithm,
        "cohort_digest": digest,
        "revision": revision,
        "kappa": kappa,
        "kappa_interval": interval,
        "threshold": threshold,
        "pass": passed,
        "method": method,
        "positive_agreement": positive_agreement,
        "negative_agreement": negative_agreement,
        "weighted_error_rate": weighted,
        "weighted_false_merge_disagreement_rate": false_rate,
        "weighted_missed_merge_disagreement_rate": missed_rate,
        "strata": {
            s: {"denominator": denominators[s], "rated": len(e), "errors": sum(e)}
            for s, e in errors.items()
        },
        "very_similar": similar,
        "round": round_number,
        "next_action": "proceed"
        if passed
        else "revise_rubric_and_rerun"
        if round_number == 1
        else "owner_escalation",
    }


def fixture_round(output: Path) -> dict[str, Any]:
    """Create and import a synthetic blind rating round without model weights."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    hashes = {
        f"image-{i}": (value, "a" if i % 2 else "b")
        for i, value in enumerate(random.Random(42).sample(range(4096), 128))
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
        "edges": sample_edges(hashes),
        "features": feature_neighbors(paths, extractor),
        "feature_provenance": {"kind": "synthetic", "version": "fixture-v1"},
        "residual": residual_sample(
            {"fixture": {f"image-{i}": i for i in range(220)}},
            {"fixture": {f"image-{i}": i for i in range(220, 230)}},
            extractor,
        ),
        "recall": synthetic_recall(images),
    }
    (output / "samples.json").write_text(
        json.dumps(samples, indent=2) + "\n", encoding="utf-8"
    )
    cohort = prepare_cohort(samples["edges"])
    rows, denominators = cohort["pairs"], cohort["denominators"]
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
    first_failed = evaluate_ratings(rows, failed_path, denominators, round_number=1)
    failed = [
        first_failed,
        evaluate_ratings(
            rows,
            failed_path,
            denominators,
            round_number=2,
            previous=first_failed,
            revision="synthetic fixture revision",
        ),
    ]
    (output / "failed_rounds.json").write_text(
        json.dumps(failed, indent=2) + "\n", encoding="utf-8"
    )
    report = evaluate_ratings(rows, output / "synthetic_ratings.csv", denominators)
    (output / "agreement.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
