from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from jute_disease.config import BENCHMARK_CONFIG
from jute_disease.training import (
    ProgressRecord,
    StackError,
    configure_runtime,
    load_benchmark,
    training_callbacks,
    verified_progress,
    verify_stack,
)


def test_runtime_config_enables_tf32_and_turns_off_cudnn_benchmark() -> None:
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    old_benchmark = torch.backends.cudnn.benchmark
    old_cudnn_deterministic = torch.backends.cudnn.deterministic
    before = torch.are_deterministic_algorithms_enabled()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        backend = configure_runtime()

        assert backend == {
            "tf32_matmul": True,
            "cudnn_tf32": True,
            "cudnn_benchmark": False,
        }
        assert torch.backends.cudnn.deterministic is True
        assert torch.are_deterministic_algorithms_enabled() is before
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        torch.backends.cudnn.benchmark = old_benchmark
        torch.backends.cudnn.deterministic = old_cudnn_deterministic


def test_resumable_callbacks_record_only_after_checkpoint_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"

    def save_last(callback, trainer, module) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "last.ckpt").write_bytes(b"post-callback-checkpoint")

    monkeypatch.setattr(ModelCheckpoint, "on_validation_end", save_last)
    callbacks = training_callbacks(
        load_benchmark(BENCHMARK_CONFIG).recipe,
        checkpoint_dir,
        resumable=True,
        attempt="attempt-2",
    )
    checkpoint = next(
        callback for callback in callbacks if isinstance(callback, ModelCheckpoint)
    )

    assert isinstance(checkpoint, ProgressRecord)
    checkpoint.on_validation_end(
        SimpleNamespace(current_epoch=4, sanity_checking=False), None
    )
    progress = verified_progress(checkpoint_dir)

    assert progress is not None
    assert progress["attempt"] == "attempt-2"
    assert progress["epochs_completed"] == 5


def test_resumable_callbacks_need_an_attempt_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="attempt name"):
        training_callbacks(
            load_benchmark(BENCHMARK_CONFIG).recipe,
            tmp_path / "checkpoints",
            resumable=True,
        )


def test_cuda_required_rejects_cpu_accelerator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(StackError, match="selected accelerator is not CUDA"):
        verify_stack(require_cuda=True, accelerator="cpu")
