"""The shared CNN fit on tiny frozen fixtures: CPU, resnet18, 64 px, no test pixels."""

import csv
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from jute_disease.config import BENCHMARK_CONFIG
from jute_disease.data import DataModule, file_sha256
from jute_disease.models.dl.backbone import TimmBackbone, state_digest
from jute_disease.training import (
    DataInputs,
    FitKind,
    FitSpec,
    Recipe,
    fit_cnn,
    load_benchmark,
    training_callbacks,
)

from .conftest import Inputs


@pytest.fixture(scope="module")
def recipe() -> Recipe:
    return replace(
        load_benchmark(BENCHMARK_CONFIG).recipe,
        image_size=64,
        max_epochs=2,
        precision="32-true",
    )


@pytest.fixture(scope="module")
def data(inputs: Inputs) -> DataInputs:
    return DataInputs(
        image_root=inputs.image_root,
        target_dir=inputs.target_dir,
        sources=inputs.sources,
    )


def _target(upstream: str | None = None, accumulation: int = 2) -> FitSpec:
    return FitSpec(
        "cnn/tiny/seed-42",
        FitKind.TARGET,
        42,
        system="tiny",
        model="resnet18",
        micro_batch=8,
        accumulation=accumulation,
        upstream=upstream,
    )


def _source(stage: str) -> FitSpec:
    return FitSpec(
        f"source/{stage}",
        FitKind.SOURCE,
        42,
        stage=stage,
        model="resnet18",
        micro_batch=8,
        accumulation=1,
    )


def _fit(spec: FitSpec, recipe: Recipe, data: DataInputs, tmp_path: Path, **kw):
    return fit_cnn(spec, recipe, data, tmp_path, pretrained=False, **kw)


def test_callbacks_follow_the_declared_selection_rule(recipe: Recipe) -> None:
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    full = load_benchmark(BENCHMARK_CONFIG).recipe
    callbacks = training_callbacks(full, Path("checkpoints"))
    checkpoint = next(c for c in callbacks if isinstance(c, ModelCheckpoint))
    stopping = next(c for c in callbacks if isinstance(c, EarlyStopping))
    assert (checkpoint.monitor, checkpoint.mode, checkpoint.save_top_k) == (
        "val_loss",
        "min",
        1,
    )
    assert (stopping.monitor, stopping.mode, stopping.patience) == (
        "val_loss",
        "min",
        8,
    )
    assert full.max_epochs == 50


def test_target_fit_records_the_callback_selected_checkpoint(
    recipe: Recipe, data: DataInputs, tmp_path: Path
) -> None:
    result = _fit(_target(), recipe, data, tmp_path)
    results = result.results
    checkpoint = result.artifacts["checkpoint"]
    assert checkpoint.is_file()
    assert results["checkpoint"]["sha256"] == file_sha256(checkpoint)
    assert results["epochs_run"] == 2
    assert results["early_stopped"] is False
    assert len(list(checkpoint.parent.glob("*.ckpt"))) == 1

    with result.artifacts["metrics"].open(encoding="utf-8") as stream:
        val_losses = [
            float(row["val_loss"])
            for row in csv.DictReader(stream)
            if row.get("val_loss")
        ]
    assert len(val_losses) == 2
    assert results["checkpoint"]["val_loss"] == pytest.approx(min(val_losses))
    assert results["seconds_per_epoch"] > 0


def test_fit_never_opens_or_reports_the_test_partition(
    recipe: Recipe, data: DataInputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    original = DataModule._manifest_dataset

    def spy(self: DataModule, partition: str, train: bool):
        opened.append(partition)
        return original(self, partition, train)

    monkeypatch.setattr(DataModule, "_manifest_dataset", spy)
    result = _fit(_target(), recipe, data, tmp_path)
    assert set(opened) == {"train", "val"}
    assert result.results["test_accessed"] is False
    assert not any("test" in key for key in result.results["metrics"])
    with result.artifacts["metrics"].open(encoding="utf-8") as stream:
        assert not any("test" in name for name in csv.DictReader(stream).fieldnames)


def test_full_backbone_is_trained_and_effective_batch_is_recorded(
    recipe: Recipe, data: DataInputs, tmp_path: Path
) -> None:
    results = _fit(_target(accumulation=2), recipe, data, tmp_path).results
    assert results["effective_batch"] == 16
    assert results["accumulate_grad_batches"] == 2
    assert results["backbone"]["trainable"] == results["backbone"]["total"]
    assert results["backbone"]["updated"] == results["backbone"]["trainable"]
    assert results["compiled"] is False
    assert results["weighted_sampler"] is True


def test_source_fit_uses_the_source_partitions(
    recipe: Recipe, data: DataInputs, tmp_path: Path
) -> None:
    results = _fit(_source("plantvillage"), recipe, data, tmp_path).results
    manifest_dir, _ = data.sources["plantvillage"]
    assert results["lineage"]["manifests"]["train"] != results["lineage"][
        "manifests"
    ].get("target")
    assert results["lineage"]["stage"] == "plantvillage"
    assert results["lineage"]["upstream_fit"] is None
    assert manifest_dir.is_dir()


def test_chained_target_transfers_the_source_backbone_and_resets_the_head(
    recipe: Recipe, data: DataInputs, tmp_path: Path
) -> None:
    source = _fit(_source("plantvillage"), recipe, data, tmp_path / "source")
    upstream = source.artifacts["checkpoint"]

    result = fit_cnn(
        _target(upstream="source/plantvillage"),
        recipe,
        data,
        tmp_path / "target",
        upstream_checkpoint=upstream,
        pretrained=False,
    )
    results = result.results

    expected = TimmBackbone("resnet18", pretrained=False, checkpoint_path=str(upstream))
    assert results["backbone"]["initial_digest"] == state_digest(expected)
    assert results["lineage"]["upstream_checkpoint_sha256"] == file_sha256(upstream)
    assert results["lineage"]["upstream_fit"] == "source/plantvillage"
    assert results["transfer"]["tensors"] == expected.transfer.tensors
    assert results["head"]["fresh"] is True
    assert results["head"]["equals_upstream"] is False
    assert results["training_state"] == {"resumed": False, "optimizer_loaded": False}


def test_chained_fit_requires_its_upstream_checkpoint(
    recipe: Recipe, data: DataInputs, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="upstream"):
        _fit(_target(upstream="source/plantvillage"), recipe, data, tmp_path)


def test_bounded_smoke_is_labelled_and_limits_batches(
    recipe: Recipe, data: DataInputs, tmp_path: Path
) -> None:
    results = _fit(
        _target(),
        recipe,
        data,
        tmp_path,
        limit_train_batches=2,
        limit_val_batches=1,
    ).results
    assert results["bounds"] == {"train_batches": 2, "val_batches": 1}
    assert results["bounded"] is True


def test_unbounded_fit_is_not_labelled_bounded(
    recipe: Recipe, data: DataInputs, tmp_path: Path
) -> None:
    results = _fit(_target(), replace(recipe, max_epochs=1), data, tmp_path).results
    assert results["bounded"] is False
    assert results["bounds"] == {"train_batches": None, "val_batches": None}


def test_full_epoch_size_is_recorded_so_a_bounded_run_can_be_scaled(
    recipe: Recipe, data: DataInputs, tmp_path: Path
) -> None:
    results = _fit(
        _target(), recipe, data, tmp_path, limit_train_batches=1, limit_val_batches=1
    ).results
    batches = results["epoch_batches"]
    assert batches["train"] > 1 and batches["val"] >= 1


def test_initialization_is_recorded(
    recipe: Recipe, data: DataInputs, tmp_path: Path
) -> None:
    results = _fit(_target(), replace(recipe, max_epochs=1), data, tmp_path).results
    assert results["pretrained"] is False


def test_requested_wandb_that_cannot_start_does_not_fail_the_fit(
    recipe: Recipe,
    data: DataInputs,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lightning.pytorch.loggers as loggers

    class Broken:
        def __init__(self, **kwargs: object) -> None:
            pass

        @property
        def experiment(self) -> object:
            raise RuntimeError("wandb-core did not start")

    monkeypatch.setattr(loggers, "WandbLogger", Broken)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    result = _fit(
        _target(),
        replace(recipe, max_epochs=1),
        data,
        tmp_path,
        wandb_mode="online",
        limit_train_batches=1,
        limit_val_batches=1,
    )
    assert result.results["wandb"] == "unavailable"
    assert result.artifacts["checkpoint"].is_file()


def test_stack_report_names_cuda_and_library_versions() -> None:
    from jute_disease.training import verify_stack

    report = verify_stack(require_cuda=False)
    assert report["compile_enabled"] is False
    assert report["torch"] == torch.__version__
    for key in ("timm", "lightning", "scikit-learn", "albumentationsx"):
        assert report["libraries"][key]


def test_required_cuda_fails_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from jute_disease.training import StackError, verify_stack

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(StackError, match="CUDA"):
        verify_stack(require_cuda=True)
