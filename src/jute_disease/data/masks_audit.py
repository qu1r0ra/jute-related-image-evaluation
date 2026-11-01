"""Held-out scene calibration and blind leaf-mask audit workflow."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
from PIL import Image

from jute_disease.data.masks import (
    MASK_QUALITY_RATINGS,
    MASK_RUBRIC,
    SCENE_RUBRIC,
    SCENE_TAGS,
    MaskBackend,
    MaskPrediction,
    SceneObservation,
    SceneThresholds,
    classify_scene,
    extract_scene_features,
    fit_scene_thresholds,
    save_mask_artifacts,
)
from jute_disease.evaluation import agreement_gate

AUDIT_SCHEMA = "jute-mask-audit-v1"
AUDIT_COUNT = 300
OWNER_COUNT = 120
KAPPA_THRESHOLD = 0.6
MASK_FAIL_RATE_LIMIT = 0.10
AGENT_COLUMNS = ("audit_id", "overlay_path", "mask_quality")
OWNER_COLUMNS = ("audit_id", "overlay_path", "mask_quality", "scene_tag")


@dataclass(frozen=True)
class CandidateImage:
    """One unlabeled image offered for the mask and scene audit."""

    image_id: str
    path: Path

    def __post_init__(self) -> None:
        if not self.image_id:
            raise ValueError("candidate image ID is required")
        if not Path(self.path).is_file():
            raise ValueError(f"candidate image does not exist: {self.path}")


@dataclass(frozen=True)
class FixtureMaskBackend:
    """Weight-free mask stub for an end-to-end synthetic audit fixture."""

    model: str = "fixture-green-pixel-mask-stub"
    version: str = "1"

    def predict(self, image: Image.Image) -> MaskPrediction:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        mask = (rgb[..., 1] > 100) & (rgb[..., 1] > rgb[..., 0] * 1.4)
        return MaskPrediction(
            mask,
            self.model,
            {"fixture_stub": True, "selected_component_area": int(mask.sum())},
        )


def read_candidates(source: Path) -> list[CandidateImage]:
    """Read a JSON array of image_id/path records relative to its manifest."""
    records = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("image manifest must be a nonempty JSON array")
    candidates = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"image_id", "path"}:
            raise ValueError("each manifest record needs only image_id and path")
        path = Path(record["path"])
        if not path.is_absolute():
            path = Path(source).parent / path
        candidates.append(CandidateImage(str(record["image_id"]), path.resolve()))
    ids = [candidate.image_id for candidate in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("image manifest IDs must be unique")
    return candidates


def read_development_observations(
    source: Path, backend: MaskBackend
) -> list[SceneObservation]:
    """Create 100 labeled scene observations from the held-apart dev manifest."""
    records = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != 100:
        raise ValueError("scene development manifest must contain exactly 100 images")
    observations: list[SceneObservation] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "image_id",
            "path",
            "scene_tag",
        }:
            raise ValueError(
                "development records need image_id, path and scene_tag fields"
            )
        path = Path(record["path"])
        if not path.is_absolute():
            path = Path(source).parent / path
        path = path.resolve()
        candidate = CandidateImage(str(record["image_id"]), path)
        with Image.open(candidate.path) as opened:
            image = opened.convert("RGB")
        prediction = backend.predict(image)
        features = extract_scene_features(image, prediction.mask)
        observations.append(
            SceneObservation(
                image_id=candidate.image_id,
                features=features,
                scene_tag=str(record["scene_tag"]),
                source_sha256=file_sha256(candidate.path),
            )
        )
    return observations


def fit_thresholds_from_manifest(source: Path, backend: MaskBackend) -> SceneThresholds:
    """Fit cutoffs from exactly 100 labeled development images."""
    return fit_scene_thresholds(
        read_development_observations(source, backend),
        mask_model=backend.model,
        mask_version=backend.version,
    )


def write_thresholds(path: Path, thresholds: SceneThresholds) -> None:
    """Write one new JSON calibration artifact without replacing prior work."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(thresholds.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def run_masks_audit(
    candidates: Sequence[CandidateImage],
    thresholds: SceneThresholds,
    backend: MaskBackend,
    output_dir: Path,
    *,
    seed: int = 42,
    sample_count: int = AUDIT_COUNT,
    owner_count: int = OWNER_COUNT,
) -> dict[str, Any]:
    """Select a stratified mask audit and emit blinded rating templates."""
    if sample_count < 1 or owner_count < 1 or owner_count > sample_count:
        raise ValueError("sample and owner counts must be positive and owner <= sample")
    if len(candidates) < sample_count:
        raise ValueError(
            f"audit inventory has {len(candidates)} images; {sample_count} required"
        )
    ids = [candidate.image_id for candidate in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("audit candidate IDs must be unique")
    if set(ids).intersection(thresholds.development_ids):
        raise ValueError("development IDs overlap the held-out audit inventory")
    if (
        backend.model != thresholds.mask_model
        or backend.version != thresholds.mask_version
    ):
        raise ValueError("scene thresholds were fitted with a different mask backend")

    dev_hashes = {digest for _, digest in thresholds.development_sources}
    source_hashes: dict[str, str] = {}
    hash_to_id: dict[str, str] = {}
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        source_hash = file_sha256(candidate.path)
        if source_hash in dev_hashes:
            raise ValueError(
                "development image content overlaps the held-out audit set"
            )
        if source_hash in hash_to_id:
            raise ValueError(
                "audit inventory contains duplicate image content under IDs "
                f"{hash_to_id[source_hash]!r} and {candidate.image_id!r}"
            )
        hash_to_id[source_hash] = candidate.image_id
        source_hashes[candidate.image_id] = source_hash
        with Image.open(candidate.path) as opened:
            image = opened.convert("RGB")
        prediction = backend.predict(image)
        features = extract_scene_features(image, prediction.mask)
        scene_tag = classify_scene(features, thresholds)
        strata[scene_tag].append(
            {
                "candidate": candidate,
                "source_sha256": source_hash,
                "predicted_scene": scene_tag,
                "features": {
                    "background_fraction": features.background_fraction,
                    "color_uniformity": features.color_uniformity,
                },
            }
        )

    populations = {tag: len(strata.get(tag, [])) for tag in SCENE_TAGS}
    allocations = _balanced_allocation(populations, sample_count)
    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for tag in SCENE_TAGS:
        pool = sorted(strata.get(tag, []), key=lambda row: row["candidate"].image_id)
        amount = allocations[tag]
        if amount:
            sampled.extend(rng.sample(pool, amount))
    rng.shuffle(sampled)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    artifact_dir = output_dir / "artifacts"
    artifact_dir.mkdir()
    records: list[dict[str, Any]] = []
    for index, row in enumerate(sampled, start=1):
        candidate: CandidateImage = row["candidate"]
        with Image.open(candidate.path) as opened:
            image = opened.convert("RGB")
        prediction = backend.predict(image)
        artifacts = save_mask_artifacts(
            image,
            prediction,
            artifact_dir,
            image_id=candidate.image_id,
            model_version=backend.version,
        )
        audit_id = f"MASK-{index:04d}"
        population = populations[row["predicted_scene"]]
        sampled_in_stratum = allocations[row["predicted_scene"]]
        records.append(
            {
                "audit_id": audit_id,
                "image_id": candidate.image_id,
                "source_sha256": row["source_sha256"],
                "predicted_scene": row["predicted_scene"],
                "features": row["features"],
                "overlay_path": artifacts.overlay_path.relative_to(
                    output_dir
                ).as_posix(),
                "mask_path": artifacts.mask_path.relative_to(output_dir).as_posix(),
                "metadata_path": artifacts.metadata_path.relative_to(
                    output_dir
                ).as_posix(),
                "stratum_population": population,
                "stratum_sampled": sampled_in_stratum,
                "expansion_weight": population / sampled_in_stratum,
            }
        )

    owner_records = rng.sample(records, owner_count)
    manifest = {
        "schema": AUDIT_SCHEMA,
        "fixture_mode": False,
        "seed": seed,
        "requested_sample_count": sample_count,
        "sample_count": len(records),
        "owner_count": owner_count,
        "owner_audit_ids": [row["audit_id"] for row in owner_records],
        "sampling": {
            "method": "equal allocation across nonempty predicted-scene strata, "
            "then random within stratum",
            "rare_scene_oversampling": True,
            "population_by_scene": populations,
            "sampled_by_scene": allocations,
            "fail_rate_estimator": "stratified population-weighted proportion "
            "with finite-population correction",
        },
        "mask_backend": {"model": backend.model, "version": backend.version},
        "agent_rater": None,
        "mask_rubric": MASK_RUBRIC,
        "scene_rubric": SCENE_RUBRIC,
        "scene_thresholds": thresholds.to_dict(),
        "records": records,
        "instructions": {
            "agent": "Rate each overlay independently using the mask rubric; "
            "enter pass, minor, or fail.",
            "owner": "In one blind sitting, rate mask quality and scene tag for "
            "each selected overlay. Do not view model outputs or agent ratings first.",
            "sequence": "Complete this owner sitting before opening the grouping-pair "
            "worksheet.",
        },
    }
    _write_json_new(output_dir / "audit.json", manifest)
    _write_rating_sheet(output_dir / "agent.csv", AGENT_COLUMNS, records)
    _write_rating_sheet(output_dir / "owner.csv", OWNER_COLUMNS, owner_records)
    return manifest


def evaluate_masks_audit(
    manifest: dict[str, Any],
    agent_sheet: Path,
    owner_sheet: Path,
    *,
    seed: int = 42,
    resamples: int = 10_000,
    agent_rater: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate mask agreement, weighted fail rate and scene agreement."""
    if manifest.get("schema") != AUDIT_SCHEMA:
        raise ValueError("unsupported mask audit manifest")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("mask audit manifest has no records")
    by_id = {row["audit_id"]: row for row in records}
    if len(by_id) != len(records):
        raise ValueError("audit manifest IDs must be unique")
    owner_ids = manifest.get("owner_audit_ids")
    if not isinstance(owner_ids, list) or len(set(owner_ids)) != len(owner_ids):
        raise ValueError("owner subset IDs must be unique")
    if len(owner_ids) != manifest.get("owner_count"):
        raise ValueError("owner subset size does not match audit manifest")
    if not set(owner_ids).issubset(by_id):
        raise ValueError("owner subset contains IDs outside the audit sample")

    agent = _read_rating_sheet(
        Path(agent_sheet),
        AGENT_COLUMNS,
        by_id,
        {"mask_quality": MASK_QUALITY_RATINGS},
    )
    owner = _read_rating_sheet(
        Path(owner_sheet),
        OWNER_COLUMNS,
        {key: by_id[key] for key in owner_ids},
        {"mask_quality": MASK_QUALITY_RATINGS, "scene_tag": SCENE_TAGS},
    )
    agent_mask = [agent[key]["mask_quality"] for key in owner_ids]
    owner_mask = [owner[key]["mask_quality"] for key in owner_ids]
    mask_agreement = _agreement_report(
        agent_mask,
        owner_mask,
        MASK_QUALITY_RATINGS,
        seed=seed,
        resamples=resamples,
    )

    predicted_scene = [by_id[key]["predicted_scene"] for key in owner_ids]
    owner_scene = [owner[key]["scene_tag"] for key in owner_ids]
    scene_agreement = _agreement_report(
        owner_scene,
        predicted_scene,
        SCENE_TAGS,
        seed=seed + 1,
        resamples=resamples,
    )
    confusion = _confusion_matrix(owner_scene, predicted_scene, SCENE_TAGS)
    fail_rate = _weighted_fail_rate(records, agent)
    is_fixture = bool(manifest.get("fixture_mode", False))
    if agent_rater is None and is_fixture:
        agent_rater = {"name": "synthetic-fixture-rater", "version": "1"}
    if not agent_rater or not agent_rater.get("name") or not agent_rater.get("version"):
        raise ValueError(
            "agent rater name and version are required for real audit results"
        )
    return {
        "schema": "jute-mask-audit-evaluation-v1",
        "fixture_mode": is_fixture,
        "evidence_status": (
            "synthetic fixture only; not an audit result"
            if is_fixture
            else "owner and agent ratings supplied"
        ),
        "agent_rater": agent_rater,
        "mask_quality": {
            **mask_agreement,
            "fail_rate": fail_rate,
            "gate_threshold": KAPPA_THRESHOLD,
            "fail_rate_limit": MASK_FAIL_RATE_LIMIT,
            "gate_pass": mask_agreement["lower_95"] is not None
            and mask_agreement["lower_95"] >= KAPPA_THRESHOLD
            and fail_rate["estimate"] < MASK_FAIL_RATE_LIMIT,
        },
        "scene_tag": {
            **scene_agreement,
            "gate_threshold": KAPPA_THRESHOLD,
            "gate_pass": scene_agreement["lower_95"] is not None
            and scene_agreement["lower_95"] >= KAPPA_THRESHOLD,
            "confusion_matrix": {
                "rows": "owner rating",
                "columns": "automatic scene tag",
                "labels": list(SCENE_TAGS),
                "counts": confusion,
            },
        },
        "rating_counts": {
            "agent_mask": dict(Counter(row["mask_quality"] for row in agent.values())),
            "owner_mask": dict(Counter(row["mask_quality"] for row in owner.values())),
            "owner_scene": dict(Counter(row["scene_tag"] for row in owner.values())),
        },
    }


def run_masks_fixture(output_dir: Path, *, seed: int = 42) -> dict[str, Any]:
    """Run the full calibration, sample, rating and evaluation path without weights."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    input_dir = output_dir / "fixture-inputs"
    input_dir.mkdir()
    backend = FixtureMaskBackend()

    development_records: list[dict[str, str]] = []
    for index in range(100):
        tag = _fixture_tag(index % 100)
        name = f"dev-{index:03d}.png"
        _write_fixture_image(input_dir / name, tag, index)
        development_records.append(
            {"image_id": f"dev-{index:03d}", "path": name, "scene_tag": tag}
        )
    development_path = input_dir / "development.json"
    _write_json_new(development_path, development_records)
    thresholds = fit_thresholds_from_manifest(development_path, backend)
    write_thresholds(output_dir / "scene-thresholds.json", thresholds)

    candidate_records: list[dict[str, str]] = []
    counts = {"clean-background": 280, "field": 120, "unknown": 20}
    index = 0
    for tag, amount in counts.items():
        for _local_index in range(amount):
            name = f"audit-{index:04d}.png"
            _write_fixture_image(input_dir / name, tag, index + 100)
            candidate_records.append({"image_id": f"audit-{index:04d}", "path": name})
            index += 1
    inventory_path = input_dir / "audit-inventory.json"
    _write_json_new(inventory_path, candidate_records)
    manifest = run_masks_audit(
        read_candidates(inventory_path),
        thresholds,
        backend,
        output_dir / "audit",
        seed=seed,
        sample_count=AUDIT_COUNT,
        owner_count=OWNER_COUNT,
    )
    manifest["fixture_mode"] = True
    manifest["fixture_note"] = (
        "All images and ratings were generated deterministically by local stubs."
    )
    (output_dir / "audit" / "audit.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _fill_fixture_ratings(output_dir / "audit", manifest, seed)
    report = evaluate_masks_audit(
        manifest,
        output_dir / "audit" / "agent.csv",
        output_dir / "audit" / "owner.csv",
        seed=seed,
        agent_rater={"name": "synthetic-fixture-rater", "version": "1"},
    )
    _write_json_new(output_dir / "evaluation.json", report)
    return report


def file_sha256(path: Path) -> str:
    """Hash a source file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _balanced_allocation(populations: dict[str, int], target: int) -> dict[str, int]:
    if target > sum(populations.values()):
        raise ValueError("sample exceeds audit population")
    nonempty_strata = sum(population > 0 for population in populations.values())
    if target < nonempty_strata:
        raise ValueError("sample must include every nonempty scene stratum")
    allocation = dict.fromkeys(SCENE_TAGS, 0)
    remaining = target
    while remaining:
        moved = False
        for tag in SCENE_TAGS:
            if allocation[tag] < populations[tag]:
                allocation[tag] += 1
                remaining -= 1
                moved = True
                if remaining == 0:
                    break
        if not moved:
            raise ValueError("could not allocate the requested sample")
    return allocation


def _write_rating_sheet(
    path: Path, columns: Sequence[str], records: Iterable[dict[str, Any]]
) -> None:
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "audit_id": record["audit_id"],
                    "overlay_path": record["overlay_path"],
                    **{
                        column: ""
                        for column in columns
                        if column not in {"audit_id", "overlay_path"}
                    },
                }
            )


def _read_rating_sheet(
    path: Path,
    columns: Sequence[str],
    expected: dict[str, dict[str, Any]],
    allowed_values: dict[str, Sequence[str]],
) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise ValueError(f"{path.name} columns must be {', '.join(columns)}")
        rows = list(reader)
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        audit_id = row["audit_id"]
        if audit_id in indexed:
            raise ValueError(f"duplicate rating ID: {audit_id}")
        if audit_id not in expected:
            raise ValueError(f"unexpected rating ID: {audit_id}")
        if row["overlay_path"] != expected[audit_id]["overlay_path"]:
            raise ValueError(f"overlay path changed for rating ID {audit_id}")
        for column, categories in allowed_values.items():
            if row[column] not in categories:
                raise ValueError(
                    f"invalid {column} rating for {audit_id}: {row[column]!r}"
                )
        indexed[audit_id] = row
    if set(indexed) != set(expected):
        raise ValueError("rating IDs do not match the required audit subset")
    return indexed


def _agreement_report(
    first: Sequence[str],
    second: Sequence[str],
    categories: Sequence[str],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    if len(first) != len(second) or not first:
        raise ValueError("agreement requires nonempty paired ratings")
    valid = set(categories)
    if any(value not in valid for value in (*first, *second)):
        raise ValueError("rating contains a category outside the rubric")
    mapping = {category: index for index, category in enumerate(categories)}
    try:
        result = agreement_gate(
            [mapping[value] for value in first],
            [mapping[value] for value in second],
            threshold=KAPPA_THRESHOLD,
            resamples=resamples,
            seed=seed,
        )
    except ValueError as error:
        return {
            "kappa": None,
            "lower_95": None,
            "upper_95": None,
            "agreement_pass": False,
            "method": f"95% paired percentile bootstrap; undefined: {error}",
            "sample_size": len(first),
        }
    return {
        "kappa": result.kappa,
        "lower_95": result.interval.lower,
        "upper_95": result.interval.upper,
        "agreement_pass": bool(result.interval.lower >= KAPPA_THRESHOLD),
        "method": result.method,
        "sample_size": len(first),
    }


def _confusion_matrix(
    owner: Sequence[str], predicted: Sequence[str], labels: Sequence[str]
) -> list[list[int]]:
    matrix = [[0 for _ in labels] for _ in labels]
    positions = {label: index for index, label in enumerate(labels)}
    for owner_tag, predicted_tag in zip(owner, predicted, strict=True):
        matrix[positions[owner_tag]][positions[predicted_tag]] += 1
    return matrix


def _weighted_fail_rate(
    records: Sequence[dict[str, Any]], agent: dict[str, dict[str, str]]
) -> dict[str, Any]:
    by_stratum: dict[str, list[bool]] = defaultdict(list)
    populations: dict[str, int] = {}
    for record in records:
        tag = record["predicted_scene"]
        population = int(record["stratum_population"])
        if tag in populations and populations[tag] != population:
            raise ValueError("inconsistent audit population count within a stratum")
        populations[tag] = population
        by_stratum[tag].append(agent[record["audit_id"]]["mask_quality"] == "fail")
    total_population = sum(populations.values())
    if total_population < 1:
        raise ValueError("audit population must be nonempty")
    estimate = (
        sum(
            populations[tag] * sum(values) / len(values)
            for tag, values in by_stratum.items()
        )
        / total_population
    )
    variance_terms: list[float] = []
    variance_available = True
    for tag, values in by_stratum.items():
        population = populations[tag]
        sample_size = len(values)
        if sample_size == population:
            continue
        if sample_size < 2:
            variance_available = False
            break
        stratum_estimate = sum(values) / sample_size
        sample_variance = (
            sample_size / (sample_size - 1) * stratum_estimate * (1 - stratum_estimate)
        )
        weight = population / total_population
        variance_terms.append(
            weight**2 * (1 - sample_size / population) * sample_variance / sample_size
        )
    if variance_available:
        standard_error = sqrt(sum(variance_terms))
        radius = NormalDist().inv_cdf(0.975) * standard_error
        lower, upper = max(0.0, estimate - radius), min(1.0, estimate + radius)
    else:
        standard_error = lower = upper = None
    return {
        "estimate": estimate,
        "lower_95": lower,
        "upper_95": upper,
        "standard_error": standard_error,
        "gate_pass": estimate < MASK_FAIL_RATE_LIMIT,
        "method": "stratified expansion estimate with finite-population-corrected "
        "normal 95% interval",
        "population": total_population,
        "sampled": len(records),
        "population_by_scene": populations,
        "sampled_by_scene": {tag: len(rows) for tag, rows in by_stratum.items()},
    }


def _write_json_new(path: Path, value: Any) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _fixture_tag(index: int) -> str:
    return SCENE_TAGS[index % len(SCENE_TAGS)]


def _write_fixture_image(path: Path, tag: str, seed: int) -> None:
    rng = random.Random(seed)
    size = 32
    image = np.empty((size, size, 3), dtype=np.uint8)
    image[:] = (245, 245, 245)
    if tag == "field":
        for y in range(size):
            for x in range(size):
                value = 30 + ((x * 31 + y * 47 + seed * 13) % 190)
                image[y, x] = (value, value // 2, value // 3)
    if tag == "unknown":
        image[3:29, 3:29] = (0, 180, 0)
    else:
        left = rng.randrange(4, 8)
        top = rng.randrange(4, 8)
        image[top : top + 16, left : left + 16] = (0, 180, 0)
    image[0, 0] = (245, 245, seed % 256)
    image[0, 1] = (245, 245, (seed // 256) % 256)
    Image.fromarray(image, mode="RGB").save(path, format="PNG")


def _fill_fixture_ratings(
    output_dir: Path, manifest: dict[str, Any], seed: int
) -> None:
    agent_path = output_dir / "agent.csv"
    owner_path = output_dir / "owner.csv"
    records = manifest["records"]
    owner_ids = set(manifest["owner_audit_ids"])
    with agent_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(AGENT_COLUMNS))
        writer.writeheader()
        for index, row in enumerate(records):
            rating = (
                "fail"
                if index in {11, 63, 197}
                else "minor"
                if index % 17 == 0
                else "pass"
            )
            writer.writerow(
                {
                    "audit_id": row["audit_id"],
                    "overlay_path": row["overlay_path"],
                    "mask_quality": rating,
                }
            )
    rng = random.Random(seed + 29)
    with owner_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OWNER_COLUMNS))
        writer.writeheader()
        for row in records:
            if row["audit_id"] not in owner_ids:
                continue
            agent_index = int(row["audit_id"].split("-")[-1]) - 1
            mask_rating = (
                "fail"
                if agent_index in {11, 63, 197}
                else "minor"
                if agent_index % 17 == 0
                else "pass"
            )
            if rng.random() < 0.03:
                mask_rating = rng.choice(MASK_QUALITY_RATINGS)
            scene_tag = row["predicted_scene"]
            if rng.random() < 0.04:
                scene_tag = rng.choice([tag for tag in SCENE_TAGS if tag != scene_tag])
            writer.writerow(
                {
                    "audit_id": row["audit_id"],
                    "overlay_path": row["overlay_path"],
                    "mask_quality": mask_rating,
                    "scene_tag": scene_tag,
                }
            )
