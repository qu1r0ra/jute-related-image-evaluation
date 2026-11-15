"""Compare the scene-tag judges, then score the frozen thresholds once.

The compare step reads the owner's first-pass tags and one sheet per agent
judge. It lists the disagreements, reports each judge's agreement with the
owner, and draws the unanimous images the owner rechecks blind, and binds the frozen
thresholds by their digest. The test step classifies every held-out image
from those thresholds, scores it against the owner's final tags with each
pool weighted by its population, and writes one record at a fixed name inside
the compare directory. That record is claimed before any image is classified
and is never replaced, because the held-out test is run once.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from PIL import Image

from jute_disease.data.fingerprints import file_sha256
from jute_disease.data.masks import (
    SCENE_RUBRIC_VERSION,
    SCENE_TAGS,
    MaskBackend,
    SceneThresholds,
    classify_scene,
    extract_scene_features,
    require_colour_threshold,
)
from jute_disease.data.scene_sets import read_scene_set
from jute_disease.evaluation import agreement_gate, wilson_interval

SCENE_COMPARE_SCHEMA = "jute-scene-compare-v1"
SCENE_EVALUATION_SCHEMA = "jute-scene-evaluation-v1"
# Images per pool in the study population of 11,884.
POOL_POPULATIONS: dict[str, int] = {"A": 920, "B": 1390, "C": 9574}
RECHECK_COUNT = 30
ACCURACY_BAR = Fraction(9, 10)
CLEAN_RECALL_BAR = Fraction(8, 10)
MIN_SUPPORT = 10
COMPARE_FILE = "compare.json"
TEST_FILE = "test.json"
FIRST_PASS_FILE = "first-pass.csv"
DISAGREEMENTS_FILE = "disagreements.csv"
RECHECK_FILE = "recheck.csv"
TAG_COLUMNS = ("image_id", "scene_tag", "rubric_version")
RECHECK_COLUMNS = ("image_id", "path", "scene_tag", "rubric_version")
CLEAN, UNKNOWN = "clean-background", "unknown"
INSUFFICIENT = "insufficient evidence"
_RESERVED_JUDGE_NAMES = {"image_id", "owner", "path"}


def _canonical_digest(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "digest"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _some(ids: set[str]) -> str:
    return ", ".join(sorted(ids)[:5])


def _read_thresholds(path: Path) -> SceneThresholds:
    return SceneThresholds.from_dict(json.loads(Path(path).read_text("utf-8")))


def read_scene_tags(
    source: Path, expected: Sequence[str], *, sheet: str, image_set: str = "held-out"
) -> dict[str, str]:
    """Read one tag sheet that must tag exactly the ``expected`` images.

    The sheet is a CSV with ``image_id``, ``scene_tag`` and ``rubric_version``
    columns; a ``path`` column is allowed and ignored.

    Raises:
        ValueError: a column is missing or foreign, an image ID is missing,
            repeated or outside ``expected``, a tag is outside the rubric, or
            a row names another rubric version.
    """
    with Path(source).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        rows = list(reader)
    if not set(TAG_COLUMNS) <= columns <= set(RECHECK_COLUMNS):
        raise ValueError(
            f"{sheet} needs the columns {', '.join(TAG_COLUMNS)} "
            f"(and optionally path); got {sorted(columns)}"
        )
    tags: dict[str, str] = {}
    for row in rows:
        image_id, tag = row["image_id"], row["scene_tag"]
        if image_id in tags:
            raise ValueError(f"{sheet} tags image {image_id} more than once")
        if tag not in SCENE_TAGS:
            raise ValueError(
                f"{sheet} gives image {image_id} the tag {tag!r}, outside the "
                f"rubric ({', '.join(SCENE_TAGS)})"
            )
        if row["rubric_version"] != SCENE_RUBRIC_VERSION:
            raise ValueError(
                f"{sheet} gives image {image_id} rubric version "
                f"{row['rubric_version']!r}; {SCENE_RUBRIC_VERSION} is required"
            )
        tags[image_id] = tag
    foreign = set(tags) - set(expected)
    if foreign:
        raise ValueError(
            f"{sheet} tags {len(foreign)} image(s) not in the {image_set} set: "
            f"{_some(foreign)}"
        )
    missing = set(expected) - set(tags)
    if missing:
        raise ValueError(
            f"{sheet} is missing {len(missing)} image(s) of the {image_set} set: "
            f"{_some(missing)}"
        )
    return tags


def _judge_agreement(
    owner: Sequence[str], judge: Sequence[str], *, seed: int, resamples: int
) -> dict[str, Any]:
    index = {tag: position for position, tag in enumerate(SCENE_TAGS)}
    report: dict[str, Any] = {
        "sample_size": len(owner),
        "agreed": sum(a == b for a, b in zip(owner, judge, strict=True)),
    }
    try:
        result = agreement_gate(
            [index[tag] for tag in owner],
            [index[tag] for tag in judge],
            resamples=resamples,
            seed=seed,
        )
    except ValueError as error:
        return {
            **report,
            "kappa": None,
            "lower_95": None,
            "upper_95": None,
            "method": f"95% paired percentile bootstrap; undefined: {error}",
        }
    return {
        **report,
        "kappa": result.kappa,
        "lower_95": result.interval.lower,
        "upper_95": result.interval.upper,
        "method": result.method,
    }


def _csv_text(columns: Sequence[str], rows: Sequence[Mapping[str, str]]) -> str:
    lines = [",".join(columns)]
    lines.extend(",".join(row[column] for column in columns) for row in rows)
    return "\n".join(lines) + "\n"


def compare_scene_tags(
    held_out: Path,
    owner: Path,
    judges: Mapping[str, Path],
    thresholds: Path,
    output: Path,
    *,
    seed: int,
    recheck_count: int = RECHECK_COUNT,
    resamples: int = 10_000,
) -> dict[str, Any]:
    """Write the disagreements, each judge's kappa and the recheck set.

    ``output`` becomes a directory holding ``compare.json``, the first-pass
    tags of the owner and every judge (``first-pass.csv``), the images where
    any judge differs from the owner (``disagreements.csv``), and a blank
    sheet of at most ``recheck_count`` images every judge agreed with the
    owner on (``recheck.csv``), drawn with ``seed``. ``compare.json`` records
    the digest of ``thresholds``, the frozen scene thresholds, so the test
    step scores those thresholds and no others.

    Raises:
        ValueError: no judge is given, a judge name is unusable, a tag sheet
            does not tag exactly the held-out images within the rubric, or a
            held-out image is in the development set of the thresholds.
        FileExistsError: ``output`` already exists.
    """
    output = Path(output)
    if not judges:
        raise ValueError("the compare step needs at least one judge sheet")
    for name in judges:
        if not name or name in _RESERVED_JUDGE_NAMES or "," in name:
            raise ValueError(f"unusable judge name: {name!r}")
    if recheck_count < 1:
        raise ValueError("the recheck count must be positive")
    if output.exists():
        raise FileExistsError(f"scene comparison output already exists: {output}")
    records = read_scene_set(held_out)
    ids = sorted(record["image_id"] for record in records)
    paths = {record["image_id"]: record["path"] for record in records}
    shared = set(ids) & set(_read_thresholds(thresholds).development_ids)
    if shared:
        raise ValueError(
            f"{len(shared)} held-out image(s) are in the development set the "
            f"thresholds were fitted on: {_some(shared)}"
        )
    first = read_scene_tags(owner, ids, sheet="the owner's first-pass sheet")
    judged = {
        name: read_scene_tags(path, ids, sheet=f"judge {name}'s sheet")
        for name, path in judges.items()
    }
    names = sorted(judged)
    rows = [
        {
            "image_id": image_id,
            "owner": first[image_id],
            **{name: judged[name][image_id] for name in names},
        }
        for image_id in ids
    ]
    differing = [
        row for row in rows if any(row[name] != row["owner"] for name in names)
    ]
    disagreements = [row["image_id"] for row in differing]
    unanimous = sorted(set(ids) - set(disagreements))
    recheck = random.Random(f"scene-recheck:{seed}").sample(
        unanimous, min(recheck_count, len(unanimous))
    )
    report: dict[str, Any] = {
        "schema": SCENE_COMPARE_SCHEMA,
        "rubric_version": SCENE_RUBRIC_VERSION,
        "seed": seed,
        "images": len(ids),
        "inputs": {
            "held_out": file_sha256(held_out),
            "owner_first_pass": file_sha256(owner),
            "judges": {name: file_sha256(judges[name]) for name in names},
            "thresholds": file_sha256(thresholds),
        },
        "judges": {
            name: _judge_agreement(
                [first[image_id] for image_id in ids],
                [judged[name][image_id] for image_id in ids],
                seed=seed,
                resamples=resamples,
            )
            for name in names
        },
        "disagreements": disagreements,
        "recheck": {
            "method": "seeded simple random sample, without replacement, of "
            "the images every judge agreed with the owner's first tag on",
            "requested": recheck_count,
            "unanimous_images": len(unanimous),
            "image_ids": recheck,
        },
    }
    columns = ("image_id", "owner", *names)
    texts = {
        FIRST_PASS_FILE: _csv_text(columns, rows),
        DISAGREEMENTS_FILE: _csv_text(columns, differing),
        RECHECK_FILE: _csv_text(
            RECHECK_COLUMNS,
            [
                {
                    "image_id": image_id,
                    "path": paths[image_id],
                    "scene_tag": "",
                    "rubric_version": SCENE_RUBRIC_VERSION,
                }
                for image_id in recheck
            ],
        ),
        COMPARE_FILE: json.dumps(report, indent=2, sort_keys=True) + "\n",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for name, text in texts.items():
            with (staging / name).open("w", encoding="utf-8", newline="") as handle:
                handle.write(text)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return report


def _counted(correct: int, total: int) -> dict[str, Any]:
    interval = wilson_interval(correct, total)
    return {
        "correct": correct,
        "total": total,
        "estimate": interval.estimate,
        "lower_95": interval.lower,
        "upper_95": interval.upper,
    }


def _weighted(weights: Sequence[Fraction], hits: Sequence[bool]) -> dict[str, Any]:
    """A weighted proportion with a Wilson interval at the Kish effective size."""
    support = sum(weights, Fraction(0))
    correct = sum((w for w, hit in zip(weights, hits, strict=True) if hit), Fraction(0))
    exact = correct / support
    effective = float(support**2 / sum((w * w for w in weights), Fraction(0)))
    interval = wilson_interval(float(exact) * effective, effective)
    return {
        "exact": exact,
        "estimate": float(exact),
        "lower_95": interval.lower,
        "upper_95": interval.upper,
        "weighted_correct": float(correct),
        "weighted_support": float(support),
        "effective_sample_size": effective,
    }


def score_scene_test(
    rows: Sequence[tuple[str, str, str]],
    populations: Mapping[str, int] = POOL_POPULATIONS,
) -> dict[str, Any]:
    """Score ``(pool, owner tag, predicted tag)`` rows against the pass bars.

    Each image is weighted by its pool's population over the pool's images in
    ``rows``. Pooled accuracy is then the population-share-weighted mean of
    the pool accuracies, and pooled recall is a ratio of weighted confusion
    counts. Pooled intervals are Wilson intervals at the Kish effective
    sample size, (sum of weights)² / (sum of squared weights).

    Raises:
        ValueError: the rows do not cover exactly the pools of
            ``populations``, or a tag is outside the rubric.
    """
    pools = sorted(populations)
    sizes = {pool: sum(row[0] == pool for row in rows) for pool in pools}
    if {row[0] for row in rows} != set(pools) or not all(sizes.values()):
        raise ValueError(f"the held-out set must hold images of the pools {pools}")
    if any(
        tag not in SCENE_TAGS
        for _, owner, predicted in rows
        for tag in (owner, predicted)
    ):
        raise ValueError("a scene tag is outside the rubric")
    total = sum(populations.values())
    weight = {pool: Fraction(populations[pool], sizes[pool]) for pool in pools}

    def recall(tag: str, pool: str | None = None) -> tuple[int, int]:
        chosen = [
            predicted == tag
            for row_pool, owner, predicted in rows
            if owner == tag and pool in (None, row_pool)
        ]
        return sum(chosen), len(chosen)

    def weighted_recall(tag: str) -> dict[str, Any]:
        chosen = [row for row in rows if row[1] == tag]
        return _weighted(
            [weight[row[0]] for row in chosen], [row[2] == tag for row in chosen]
        )

    def confusion(pool: str | None) -> dict[str, dict[str, float]]:
        matrix = {owner: dict.fromkeys(SCENE_TAGS, Fraction(0)) for owner in SCENE_TAGS}
        for row_pool, owner, predicted in rows:
            if pool in (None, row_pool):
                matrix[owner][predicted] += 1 if pool else weight[row_pool]
        return {
            owner: {tag: float(count) for tag, count in counts.items()}
            for owner, counts in matrix.items()
        }

    by_pool: dict[str, Any] = {}
    for pool in pools:
        correct = sum(owner == predicted for p, owner, predicted in rows if p == pool)
        found, clean = recall(CLEAN, pool)
        if clean >= MIN_SUPPORT:
            clean_recall = {"status": "reported", **_counted(found, clean)}
        else:
            clean_recall = {
                "status": INSUFFICIENT if clean else "none",
                "correct": found,
                "total": clean,
            }
        by_pool[pool] = {
            "images": sizes[pool],
            "population": populations[pool],
            "share": populations[pool] / total,
            "accuracy": _counted(correct, sizes[pool]),
            "clean_recall": clean_recall,
            "confusion": confusion(pool),
        }

    accuracy = _weighted(
        [weight[row[0]] for row in rows], [row[1] == row[2] for row in rows]
    )
    accuracy_met = accuracy.pop("exact") >= ACCURACY_BAR
    recall_met = False
    pooled_recall: dict[str, Any] = {"status": "none"}
    if recall(CLEAN)[1]:
        pooled_recall = {"status": "reported", **weighted_recall(CLEAN)}
        recall_met = pooled_recall.pop("exact") >= CLEAN_RECALL_BAR
    support = recall(UNKNOWN)[1]
    unknown: dict[str, Any] = {"status": INSUFFICIENT, "support": support}
    if support >= MIN_SUPPORT:
        unknown = {
            "status": "reported",
            "support": support,
            "support_by_pool": {pool: recall(UNKNOWN, pool)[1] for pool in pools},
            **weighted_recall(UNKNOWN),
        }
        del unknown["exact"]
    return {
        "method": "each image weighs its pool's population over the pool's "
        "held-out images; pooled accuracy and recall are ratios of weighted "
        "counts; pool intervals are Wilson 95% intervals, pooled intervals "
        "Wilson 95% at the Kish effective sample size; the pass bars read "
        "the pooled point estimates",
        "bars": {
            "accuracy": float(ACCURACY_BAR),
            "clean_recall": float(CLEAN_RECALL_BAR),
        },
        "minimum_support": MIN_SUPPORT,
        "pools": by_pool,
        "pooled": {
            "accuracy": accuracy,
            "clean_recall": pooled_recall,
            "confusion": confusion(None),
        },
        "unknown": unknown,
        "accuracy_met": accuracy_met,
        "clean_recall_met": recall_met,
        "passed": accuracy_met and recall_met,
    }


def _check_images(records: Sequence[Mapping[str, str]]) -> None:
    """Check that each held-out image is present and matches its ID."""
    for record in records:
        path = Path(record["path"])
        if not path.is_file():
            raise ValueError(f"held-out image is missing: {path}")
        if file_sha256(path) != record["image_id"]:
            raise ValueError(f"held-out image does not match its image ID: {path}")


def _classify(
    records: Sequence[Mapping[str, str]],
    thresholds: SceneThresholds,
    backend: MaskBackend,
) -> dict[str, dict[str, Any]]:
    predicted: dict[str, dict[str, Any]] = {}
    for record in records:
        with Image.open(record["path"]) as opened:
            image = opened.convert("RGB")
        features = extract_scene_features(image, backend.predict(image).mask)
        predicted[record["image_id"]] = {
            "predicted": classify_scene(features, thresholds),
            "background_fraction": features.background_fraction,
            "color_uniformity": features.color_uniformity,
        }
    return predicted


def run_scene_test(
    held_out: Path,
    thresholds: Path,
    compare: Path,
    owner: Path,
    final: Path,
    backend: MaskBackend,
    *,
    recheck: Path | None = None,
) -> dict[str, Any]:
    """Score the frozen thresholds on the held-out set and write the one record.

    ``compare`` is the directory the compare step wrote; the held-out set,
    the owner's first-pass sheet and the thresholds must be the files it
    read. ``final`` tags every held-out image and may differ from the first
    pass only on a listed disagreement. ``recheck`` is the filled
    ``recheck.csv``.

    The record is ``test.json`` inside ``compare``. Once every input is
    checked, that file is created exclusively, before any image is
    classified, and filled when the scores are known. A run that stops in
    between leaves the file empty, and it still refuses every later run.

    Raises:
        FileExistsError: the record exists; the held-out test is run once.
        ValueError: an input differs from what the compare step read, a tag
            sheet is incomplete or outside the rubric, a final tag changed on
            an image that was not a disagreement, a held-out image is missing
            or altered, or the thresholds were fitted with another mask
            backend.
    """
    output = Path(compare) / TEST_FILE
    refusal = f"the held-out scene test is run once and its record exists: {output}"
    if output.exists():
        raise FileExistsError(refusal)
    compare_file = Path(compare) / COMPARE_FILE
    compared = json.loads(compare_file.read_text(encoding="utf-8"))
    if compared.get("schema") != SCENE_COMPARE_SCHEMA:
        raise ValueError(f"{compare_file} is not a scene comparison record")
    if file_sha256(held_out) != compared["inputs"]["held_out"]:
        raise ValueError("the held-out set differs from the one the compare step read")
    if file_sha256(owner) != compared["inputs"]["owner_first_pass"]:
        raise ValueError(
            "the owner's first-pass tags differ from the ones the compare step read"
        )
    if file_sha256(thresholds) != compared["inputs"]["thresholds"]:
        raise ValueError("the thresholds differ from the ones the compare step read")
    records = read_scene_set(held_out)
    ids = sorted(record["image_id"] for record in records)
    first = read_scene_tags(owner, ids, sheet="the owner's first-pass sheet")
    final_tags = read_scene_tags(final, ids, sheet="the final tag sheet")
    changed = [image_id for image_id in ids if final_tags[image_id] != first[image_id]]
    unlisted = set(changed) - set(compared["disagreements"])
    if unlisted:
        raise ValueError(
            f"{len(unlisted)} final tag(s) differ from the first pass on an image "
            f"that is not a listed disagreement: {_some(unlisted)}"
        )
    rechecked: dict[str, Any] = {"status": "not supplied"}
    if recheck is not None:
        recheck_ids = compared["recheck"]["image_ids"]
        again = read_scene_tags(
            recheck, recheck_ids, sheet="the recheck sheet", image_set="recheck"
        )
        moved = [
            {"image_id": image_id, "first": first[image_id], "recheck": again[image_id]}
            for image_id in recheck_ids
            if again[image_id] != first[image_id]
        ]
        agreed = len(recheck_ids) - len(moved)
        rechecked = {
            "status": "reported",
            "images": len(recheck_ids),
            "agreed": agreed,
            "agreement": agreed / len(recheck_ids) if recheck_ids else None,
            "changed": moved,
        }

    frozen = _read_thresholds(thresholds)
    require_colour_threshold(backend.model, backend.version)
    if (backend.model, backend.version) != (frozen.mask_model, frozen.mask_version):
        raise ValueError("scene thresholds were fitted with a different mask backend")
    _check_images(records)
    input_digests = {
        "held_out": compared["inputs"]["held_out"],
        "thresholds": compared["inputs"]["thresholds"],
        "compare": file_sha256(compare_file),
        "owner_first_pass": compared["inputs"]["owner_first_pass"],
        "judges": compared["inputs"]["judges"],
        "final_tags": file_sha256(final),
        "recheck_tags": None if recheck is None else file_sha256(recheck),
    }
    try:
        claim = output.open("x", encoding="utf-8", newline="\n")
    except FileExistsError:
        raise FileExistsError(refusal) from None
    with claim:
        predicted = _classify(records, frozen, backend)
        pools = {record["image_id"]: record["pool"] for record in records}
        scores = score_scene_test(
            [
                (
                    pools[image_id],
                    final_tags[image_id],
                    predicted[image_id]["predicted"],
                )
                for image_id in ids
            ]
        )
        record: dict[str, Any] = {
            "schema": SCENE_EVALUATION_SCHEMA,
            "rubric_version": SCENE_RUBRIC_VERSION,
            "inputs": input_digests,
            "mask_backend": {"model": backend.model, "version": backend.version},
            **scores,
            "final_tags": {
                "changed_from_first_pass": changed,
                "disagreements_reviewed": len(compared["disagreements"]),
            },
            "recheck": rechecked,
            "images": [
                {
                    "image_id": image_id,
                    "pool": pools[image_id],
                    "owner_final": final_tags[image_id],
                    **predicted[image_id],
                }
                for image_id in ids
            ],
        }
        record["digest"] = _canonical_digest(record)
        claim.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record
