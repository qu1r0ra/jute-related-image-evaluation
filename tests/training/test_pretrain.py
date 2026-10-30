"""Source pretraining consumes only verified, overlap-controlled manifests."""

import json
from dataclasses import replace
from pathlib import Path
from shutil import copyfile

import pytest

from jute_disease.config import CURATION_RULES_PATH
from jute_disease.data import (
    PLANTDOC,
    SOURCES,
    TARGET_DIR,
    DataModule,
    FreezeBlockedError,
    SourceMismatchError,
    SyntheticPlan,
    build_inventory,
    freeze_target,
    intake_source,
    inventory_digest,
    load_manifest,
    make_synthetic_data,
    observed_counts,
)
from jute_disease.training import pretrain_main, source_data_module


@pytest.fixture(scope="module")
def rules_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rules = json.loads(CURATION_RULES_PATH.read_text(encoding="utf-8"))
    rules["min_support"] = 2
    path = tmp_path_factory.mktemp("pretrain-rules") / "rules.json"
    path.write_text(json.dumps(rules), encoding="utf-8")
    return path


def _freeze_target(synthetic: SyntheticPlan, rules_path: Path, output: Path) -> None:
    freeze_target(
        output,
        image_root=synthetic.root,
        rules_path=rules_path,
        reviews_path=None,
        workers=2,
    )


def test_source_data_module_rejects_changed_content_and_target_overlap(
    synthetic: SyntheticPlan,
    rules_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "freeze"
    _freeze_target(synthetic, rules_path, output)
    target = load_manifest(output / TARGET_DIR)
    package = tmp_path / "package"
    make_synthetic_data(package / "train", per_class=20, labels=("a", "b"), seed=100)
    make_synthetic_data(package / "test", per_class=10, labels=("a", "b"), seed=200)
    for duplicate in (package / "test").rglob("*bytecopy.png"):
        duplicate.unlink()
    overlap_path = next((package / "train" / "a").glob("*.png"))
    copyfile(synthetic.root / target.partition("val")[0].path, overlap_path)
    records = build_inventory(package, workers=2)
    spec = replace(
        PLANTDOC,
        expected=tuple(sorted(observed_counts(records).items())),
        expected_inventory_digest=inventory_digest(records),
    )
    monkeypatch.setitem(SOURCES, "plantdoc", spec)
    intake_source(
        "plantdoc",
        package,
        output,
        package_version=7,
        package_digest=inventory_digest(records),
    )

    with pytest.raises(FreezeBlockedError, match="target_manifest_dir"):
        DataModule(
            manifest_dir=output / "source_plantdoc",
            image_root=package,
            batch_size=4,
            seed=42,
        ).setup("fit")

    data_module = source_data_module(
        output / "source_plantdoc", package, output / TARGET_DIR, 4, 42
    )
    observed_paths = {
        entry.path
        for dataset in (data_module.jute_train, data_module.jute_val)
        for entry in dataset.entries
    }
    overlap_relative = overlap_path.relative_to(package).as_posix()
    assert overlap_relative not in observed_paths

    monkeypatch.setitem(SOURCES, "plantdoc", replace(spec, version=6))
    with pytest.raises(FreezeBlockedError, match="pinned source"):
        source_data_module(
            output / "source_plantdoc", package, output / TARGET_DIR, 4, 42
        )
    monkeypatch.setitem(SOURCES, "plantdoc", spec)

    retained = data_module.jute_train.entries[0]
    (package / retained.path).write_bytes(b"changed")
    with pytest.raises(SourceMismatchError, match="Source content changed"):
        source_data_module(
            output / "source_plantdoc", package, output / TARGET_DIR, 4, 42
        )


def test_source_pretraining_blocks_missing_owner_authorization(
    synthetic: SyntheticPlan,
    rules_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "freeze"
    _freeze_target(synthetic, rules_path, output)
    package = tmp_path / "package"
    make_synthetic_data(package / "train", per_class=20, labels=("a", "b"), seed=300)
    make_synthetic_data(package / "test", per_class=10, labels=("a", "b"), seed=400)
    for duplicate in (package / "test").rglob("*bytecopy.png"):
        duplicate.unlink()
    records = build_inventory(package, workers=2)
    spec = replace(
        PLANTDOC,
        expected=tuple(sorted(observed_counts(records).items())),
        expected_inventory_digest=inventory_digest(records),
        owner_internal_use_authorized=False,
        owner_authorization_record="",
    )
    monkeypatch.setitem(SOURCES, "plantdoc", spec)
    intake_source(
        "plantdoc",
        package,
        output,
        package_version=7,
        package_digest=inventory_digest(records),
    )

    with pytest.raises(FreezeBlockedError, match="Owner authorization"):
        source_data_module(
            output / "source_plantdoc", package, output / TARGET_DIR, 4, 42
        )


def test_pretrain_command_converts_missing_source_inputs_to_usage_errors() -> None:
    with pytest.raises(SystemExit) as excinfo:
        pretrain_main(
            [
                "--manifest_dir",
                "manifest",
                "--image_root",
                "images",
                "--target_manifest_dir",
                "target",
                "--output_path",
                "output.ckpt",
            ]
        )
    assert excinfo.value.code == 2
