"""Frozen tiny fixtures for end-to-end training and queue tests.

The target is the structured synthetic dataset frozen with the real command; the
sources are two small synthetic packages registered under the pinned names, so
every training path runs through verified manifests without touching real data.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from jute_disease.config import CURATION_RULES_PATH
from jute_disease.data import (
    PLANTDOC,
    PLANTVILLAGE,
    SOURCES,
    TARGET_DIR,
    SyntheticPlan,
    build_inventory,
    freeze_target,
    intake_source,
    inventory_digest,
    make_synthetic_data,
    observed_counts,
)


@dataclass(frozen=True)
class Inputs:
    """Frozen target and verified source packages on disk."""

    image_root: Path
    freeze_dir: Path
    target_dir: Path
    sources: dict[str, tuple[Path, Path]]


def _register_source(
    name: str, base_spec: object, package: Path, freeze_dir: Path
) -> None:
    records = build_inventory(package, workers=2)
    spec = replace(
        base_spec,
        expected=tuple(sorted(observed_counts(records).items())),
        expected_inventory_digest=inventory_digest(records),
    )
    SOURCES[name] = spec
    intake_source(
        name,
        package,
        freeze_dir,
        package_version=spec.version,
        package_digest=inventory_digest(records),
    )


@pytest.fixture(scope="session")
def inputs(
    synthetic: SyntheticPlan, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Inputs]:
    base = tmp_path_factory.mktemp("training-inputs")
    rules = json.loads(CURATION_RULES_PATH.read_text(encoding="utf-8"))
    rules["min_support"] = 2
    rules_path = base / "rules.json"
    rules_path.write_text(json.dumps(rules), encoding="utf-8")
    freeze_dir = base / "freeze"
    freeze_target(
        freeze_dir,
        image_root=synthetic.root,
        rules_path=rules_path,
        reviews_path=None,
        workers=2,
    )
    saved = dict(SOURCES)
    sources: dict[str, tuple[Path, Path]] = {}
    try:
        for name, base_spec, seed in (
            ("plantvillage", PLANTVILLAGE, 100),
            ("plantdoc", PLANTDOC, 300),
        ):
            package = base / f"package_{name}"
            make_synthetic_data(
                package / "train",
                per_class=20,
                size=64,
                labels=("a", "b", "c"),
                seed=seed,
            )
            make_synthetic_data(
                package / "test",
                per_class=10,
                size=64,
                labels=("a", "b", "c"),
                seed=seed + 1,
            )
            for duplicate in (package / "test").rglob("*bytecopy.png"):
                duplicate.unlink()
            _register_source(name, base_spec, package, freeze_dir)
            sources[name] = (freeze_dir / f"source_{name}", package)
        yield Inputs(
            image_root=synthetic.root,
            freeze_dir=freeze_dir,
            target_dir=freeze_dir / TARGET_DIR,
            sources=sources,
        )
    finally:
        SOURCES.clear()
        SOURCES.update(saved)
