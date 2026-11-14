"""Source screening, the equal Healthy image count and the strict View 2 subset, via the CLI.

A small structured target is frozen with the real command (dhash-v2 grouping,
screened against two source packages that contain planted copies of target
images), then the sources and the policy matrix are produced with the real
commands. Nothing here reads data outside ``tmp_path``.
"""

import json
import shutil
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from jute_disease.cli import main
from jute_disease.config import CURATION_RULES_PATH
from jute_disease.data import (
    PLANTDOC,
    PLANTVILLAGE,
    SOURCES,
    build_inventory,
    inventory_digest,
    load_manifest,
    make_synthetic_data,
    observed_counts,
)
from jute_disease.data.grouping import flip_canonical_dhash
from jute_disease.data.inventory import read_inventory
from jute_disease.data.manifest import load_complete_manifest

from .gate_records import gate_record

LABELS = ("Healthy", "Mosaic")
FAMILIES_PER_LABEL = 24
HEALTHY_CAP = 12
THRESHOLD = 6
AUDIT_LIMIT = 16
POLICIES = ("image-random", "exact-copy", "near-duplicate")


def _field(rng: np.random.Generator, size: int = 64) -> Image.Image:
    coarse = rng.integers(0, 256, (6, 6, 3), dtype=np.uint8)
    return Image.fromarray(coarse).resize((size, size), Image.Resampling.BICUBIC)


def _shifted(image: Image.Image, delta: int = 2) -> Image.Image:
    pixels = np.asarray(image, dtype=np.int16) + delta
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def _distance(a: Image.Image, b: Image.Image) -> int:
    return (flip_canonical_dhash(a) ^ flip_canonical_dhash(b)).bit_count()


def _twin(base: Image.Image, rng: np.random.Generator) -> Image.Image:
    """An image whose dHash sits in the audit band of ``base``."""
    pixels = np.asarray(base, dtype=np.float64)
    for _ in range(200):
        other = np.asarray(_field(rng), dtype=np.float64)
        for alpha in np.arange(0.95, 0.4, -0.05):
            mixed = np.clip(alpha * pixels + (1 - alpha) * other, 0, 255)
            candidate = Image.fromarray(mixed.astype(np.uint8))
            if THRESHOLD < _distance(base, candidate) <= AUDIT_LIMIT:
                return candidate
    raise AssertionError("no audit-band twin found")


def _build_target(root: Path) -> None:
    """Families of one to two images per class, with audit-band twin families."""
    rng = np.random.default_rng(7)
    for label in LABELS:
        folder = root / label
        folder.mkdir(parents=True)
        for index in range(FAMILIES_PER_LABEL):
            stem = f"{label}_f{index:02}"
            base = _field(rng)
            base.save(folder / f"{stem}_a.png")
            if index % 3 == 1:
                _shifted(base).save(folder / f"{stem}_b.png")
            if index % 3 == 2:
                shutil.copyfile(folder / f"{stem}_a.png", folder / f"{stem}_b.png")
            if index % 4 == 0:
                _twin(base, rng).save(folder / f"{label}_t{index:02}.png")


def _build_package(
    package: Path, target: Path, seed: int, exact: list[str], near: list[str]
) -> None:
    make_synthetic_data(
        package / "train", per_class=20, size=64, labels=("a", "b", "c"), seed=seed
    )
    make_synthetic_data(
        package / "test", per_class=10, size=64, labels=("a", "b", "c"), seed=seed + 1
    )
    for duplicate in (package / "test").rglob("*bytecopy.png"):
        duplicate.unlink()
    folder = package / "train" / "a"
    for name in exact:
        shutil.copyfile(target / "Healthy" / name, folder / f"planted_{name}")
    for name in near:
        with Image.open(target / "Mosaic" / name) as image:
            _shifted(image).save(folder / f"planted_near_{name}")


def _register(name: str, base_spec: object, package: Path) -> str:
    records = build_inventory(package, workers=2)
    digest = inventory_digest(records)
    SOURCES[name] = replace(
        base_spec,
        expected=tuple(sorted(observed_counts(records).items())),
        expected_inventory_digest=digest,
    )
    return digest


class Frozen:
    """Where the command-line artifacts of the module fixture live."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.target_images = base / "images"
        self.freeze = base / "freeze"
        self.packages = {"plantvillage": base / "pv", "plantdoc": base / "pd"}
        self.digests: dict[str, str] = {}

    @property
    def grouping(self) -> dict:
        return json.loads((self.freeze / "target" / "grouping.json").read_text())

    def intake(self, name: str) -> None:
        spec = SOURCES[name]
        main(
            [
                "freeze",
                "source",
                name,
                str(self.packages[name]),
                "--output",
                str(self.freeze),
                "--package-version",
                str(spec.version),
                "--package-digest",
                self.digests[name],
                "--workers",
                "2",
            ]
        )

    def report(self, name: str) -> dict:
        return json.loads((self.freeze / f"source_{name}.json").read_text())

    def policies(self, output: Path, *extra: str) -> dict:
        main(
            [
                "freeze",
                "policies",
                "--pool",
                f"C={self.freeze}",
                "--output",
                str(output),
                "--reserve-fraction",
                "0.1",
                "--train-fraction",
                "0.7",
                "--val-fraction",
                "0.15",
                "--test-fraction",
                "0.15",
                *extra,
            ]
        )
        return json.loads((output / "receipt.json").read_text())


@pytest.fixture(scope="module")
def frozen(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Frozen]:
    state = Frozen(tmp_path_factory.mktemp("policies-and-sources"))
    _build_target(state.target_images)
    _build_package(
        state.packages["plantvillage"],
        state.target_images,
        seed=100,
        exact=[f"Healthy_f{i:02}_a.png" for i in range(0, 8)],
        near=[f"Mosaic_f{i:02}_a.png" for i in range(0, 4)],
    )
    _build_package(
        state.packages["plantdoc"],
        state.target_images,
        seed=300,
        exact=[f"Healthy_f{i:02}_a.png" for i in range(8, 16)],
        near=[f"Mosaic_f{i:02}_a.png" for i in range(4, 8)],
    )
    rules = json.loads(CURATION_RULES_PATH.read_text(encoding="utf-8"))
    rules.update(min_support=2, supported_labels=list(LABELS))
    (state.base / "rules.json").write_text(json.dumps(rules), encoding="utf-8")
    (state.base / "reviews.json").write_text(
        json.dumps({"version": 1, "decisions": []}), encoding="utf-8"
    )
    (state.base / "agreement.json").write_text(
        json.dumps(
            gate_record(
                inventory_digest(build_inventory(state.target_images, workers=2)),
                version="dhash-v2",
            )
        ),
        encoding="utf-8",
    )
    saved = dict(SOURCES)
    try:
        for name, spec in (("plantvillage", PLANTVILLAGE), ("plantdoc", PLANTDOC)):
            state.digests[name] = _register(name, spec, state.packages[name])
        main(
            [
                "freeze",
                "target",
                "--image-root",
                str(state.target_images),
                "--rules",
                str(state.base / "rules.json"),
                "--reviews",
                str(state.base / "reviews.json"),
                "--output",
                str(state.freeze),
                "--workers",
                "2",
                "--grouping-version",
                "dhash-v2",
                "--pool",
                "C",
                "--agreement-results",
                str(state.base / "agreement.json"),
                "--plantvillage-root",
                str(state.packages["plantvillage"]),
                "--plantdoc-root",
                str(state.packages["plantdoc"]),
            ]
        )
        yield state
    finally:
        SOURCES.clear()
        SOURCES.update(saved)


def test_source_intake_drops_every_screened_exclusion(
    frozen: Frozen, capsys: pytest.CaptureFixture[str]
) -> None:
    target = load_complete_manifest(frozen.freeze / "target")
    train_paths = {entry.path for entry in target.partition("train")}
    planted_train = {f"Healthy/Healthy_f{i:02}_a.png" for i in range(16)} & train_paths
    # The screened copies that the held-out check alone would not catch.
    assert planted_train, "fixture must plant copies of target training images"
    for name in ("plantvillage", "plantdoc"):
        frozen.intake(name)
        exclusions = set(frozen.grouping["source_screening"][name]["exclusions"])
        assert exclusions
        manifest = load_manifest(frozen.freeze / f"source_{name}")
        report = frozen.report(name)
        kept = {entry.path for entry in manifest.entries}
        assert not exclusions & kept
        assert exclusions <= set(report["removed"])
        assert {item["path"] for item in manifest.excluded} >= exclusions
        assert kept
    capsys.readouterr()


def test_source_intake_refuses_a_package_that_was_not_screened(
    frozen: Frozen, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path / "other-package"
    make_synthetic_data(
        other / "train", per_class=20, size=64, labels=("a", "b", "c"), seed=900
    )
    make_synthetic_data(
        other / "test", per_class=10, size=64, labels=("a", "b", "c"), seed=901
    )
    for duplicate in (other / "test").rglob("*bytecopy.png"):
        duplicate.unlink()
    digest = _register_scoped(monkeypatch, "plantdoc", PLANTDOC, other)
    scratch = tmp_path / "freeze"
    shutil.copytree(frozen.freeze, scratch)
    for leftover in scratch.glob("source_plantdoc*"):
        shutil.rmtree(leftover) if leftover.is_dir() else leftover.unlink()
    with pytest.raises(SystemExit, match="screen"):
        main(
            [
                "freeze",
                "source",
                "plantdoc",
                str(other),
                "--output",
                str(scratch),
                "--package-version",
                str(SOURCES["plantdoc"].version),
                "--package-digest",
                digest,
                "--workers",
                "2",
            ]
        )
    assert not (scratch / "source_plantdoc.json").exists()


def _register_scoped(
    monkeypatch: pytest.MonkeyPatch, name: str, base_spec: object, package: Path
) -> str:
    records = build_inventory(package, workers=2)
    digest = inventory_digest(records)
    monkeypatch.setitem(
        SOURCES,
        name,
        replace(
            base_spec,
            expected=tuple(sorted(observed_counts(records).items())),
            expected_inventory_digest=digest,
        ),
    )
    return digest


def _healthy_train_images(output: Path, view: int, policy: str, draw: int) -> int:
    path = output / "C" / f"view{view}" / policy / f"draw-{draw}.json"
    payload = json.loads(path.read_text())
    return payload["image_support"]["train"]["Healthy"]


def test_healthy_cap_is_one_image_count_in_every_policy(
    frozen: Frozen, tmp_path: Path
) -> None:
    output = tmp_path / "capped"
    receipt = frozen.policies(output, "--healthy-cap", str(HEALTHY_CAP))
    assert receipt["healthy_cap"]["images"] == HEALTHY_CAP
    for draw in range(10):
        for view in (1, 2):
            counts = {
                policy: _healthy_train_images(output, view, policy, draw)
                for policy in POLICIES
            }
            assert set(counts.values()) == {HEALTHY_CAP}, (view, draw, counts)
            for policy in POLICIES:
                path = output / "C" / f"view{view}" / policy / f"draw-{draw}.json"
                payload = json.loads(path.read_text())
                unit = "path" if policy == "image-random" else "family"
                kept = {
                    row[unit]
                    for row in payload["entries"]
                    if row["label"] == "Healthy" and row["partition"] == "train"
                }
                dropped = {row[unit] for row in payload["healthy_cap_excluded"]}
                assert not kept & dropped
    again = tmp_path / "again"
    assert frozen.policies(again, "--healthy-cap", str(HEALTHY_CAP)) == receipt


def test_unreachable_healthy_cap_stops_output(frozen: Frozen, tmp_path: Path) -> None:
    output = tmp_path / "blocked"
    with pytest.raises(SystemExit, match="policy manifests blocked"):
        frozen.policies(output, "--healthy-cap", "0")
    assert not output.exists()


def test_cap_above_every_policy_still_equalises_the_counts(
    frozen: Frozen, tmp_path: Path
) -> None:
    output = tmp_path / "loose"
    receipt = frozen.policies(output)
    for draw in range(10):
        for view in (1, 2):
            counts = {}
            for policy in POLICIES:
                path = output / "C" / f"view{view}" / policy / f"draw-{draw}.json"
                payload = json.loads(path.read_text())
                kept = payload["image_support"]["train"]["Healthy"]
                counts[policy] = kept
            assert len(set(counts.values())) == 1, (view, draw, counts)
            assert max(counts.values()) <= receipt["healthy_cap"]["images"]
    assert any(
        json.loads((output / "C" / "view1" / policy / f"draw-{draw}.json").read_text())[
            "healthy_cap_excluded"
        ]
        for policy in POLICIES
        for draw in range(10)
    )


def _hashes(frozen: Frozen) -> dict[str, int]:
    records = read_inventory(frozen.freeze / "inventory.jsonl")
    hashes = {}
    for record in records:
        with Image.open(frozen.target_images / record.path) as image:
            hashes[record.path] = flip_canonical_dhash(image)
    return hashes


def test_strict_view2_subset_is_recorded_once_in_the_receipt(
    frozen: Frozen, tmp_path: Path
) -> None:
    output = tmp_path / "strict"
    receipt = frozen.policies(output, "--healthy-cap", str(HEALTHY_CAP))
    record = receipt["strict_view2_subset"]
    assert record["audit_bands"] == {
        "C": {"threshold": THRESHOLD, "max_distance": AUDIT_LIMIT}
    }
    assert record["grouping_digests"] == {"C": frozen.grouping["digest"]}
    hashes = _hashes(frozen)
    removed_somewhere = 0
    for draw in range(10):
        protected: set[str] = set()
        fit: set[str] = set()
        for policy in POLICIES:
            base = output / "C" / "view2" / policy / f"draw-{draw}"
            fit |= {
                row["path"]
                for row in json.loads(base.with_suffix(".json").read_text())["entries"]
            }
            protected_manifest = json.loads(
                base.with_suffix(".protected-test.json").read_text()
            )
            tested = {row["path"] for row in protected_manifest["entries"]}
            assert not protected or tested == protected
            protected = tested
        expected = sorted(
            path
            for path in protected
            if not any(
                THRESHOLD < (hashes[path] ^ hashes[other]).bit_count() <= AUDIT_LIMIT
                for other in fit
            )
        )
        entry = record["pools"]["C"][draw]
        assert entry["draw"] == draw
        assert entry["protected_test"] == len(protected)
        assert entry["paths"] == expected
        removed_somewhere += len(protected) - len(expected)
    assert removed_somewhere > 0
