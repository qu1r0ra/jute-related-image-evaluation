"""Validation-only pilot: one MobileNetV2 fit through the production CNN engine.

Drives ``fit_cnn`` directly on a frozen target manifest so timing, peak VRAM,
checkpoint resume and the image-size comparison exercise the same code the
queue runs. It reads only the manifest's train and validation partitions; the
held-out test partition is never opened (``allow_test`` stays off).

Usage (from the repository root):
    uv run python scripts/pilot_fit.py --attempt-dir DIR [--checkpoint-dir DIR]
        [--resume-from DIR/last.ckpt] [--max-epochs N] [--image-size PX]
        [--manifest-dir DIR] [--image-root DIR]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

from jute_disease.training.benchmark import FitKind, FitSpec, load_benchmark
from jute_disease.training.engines.dl import fit_cnn
from jute_disease.training.inputs import DataInputs

BENCHMARK = Path("configs/benchmark.json")
DEFAULT_MANIFEST = Path("artifacts/freeze/target")
DEFAULT_IMAGE_ROOT = Path("data/recovered_original_split/ml_split")
SYSTEM = "mobilenet_v2_direct"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    recipe = load_benchmark(BENCHMARK).recipe
    overrides: dict[str, int] = {}
    if args.max_epochs is not None:
        overrides["max_epochs"] = args.max_epochs
    if args.image_size is not None:
        overrides["image_size"] = args.image_size
    recipe = dataclasses.replace(recipe, **overrides)
    fit = FitSpec(
        fit_id=f"pilot/{SYSTEM}/seed-{args.seed}",
        kind=FitKind.TARGET,
        seed=args.seed,
        system=SYSTEM,
        stage="jute",
        model="mobilenetv2_100.ra_in1k",
        micro_batch=32,
        accumulation=1,
    )
    data = DataInputs(
        image_root=args.image_root,
        target_dir=args.manifest_dir,
        sources={},
    )
    started = time.time()
    result = fit_cnn(
        fit,
        recipe,
        data,
        args.attempt_dir,
        pretrained=True,
        accelerator="gpu",
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
    )
    summary = {
        "wall_seconds": round(time.time() - started, 1),
        "image_size": recipe.image_size,
        "max_epochs": recipe.max_epochs,
        "epochs_run": result.results["epochs_run"],
        "epoch_seconds": result.results["epoch_seconds"],
        "seconds_per_epoch_max": result.results["seconds_per_epoch"],
        "peak_cuda_bytes": result.results["peak_cuda_bytes"],
        "metrics": result.results["metrics"],
        "checkpoint": result.results["checkpoint"],
        "resumed": result.results["training_state"]["resumed"],
        "resumed_from": result.results["lineage"]["resumed_from"],
        "epoch_batches": result.results["epoch_batches"],
        "bounded": result.results["bounded"],
        "test_accessed": result.results["test_accessed"],
    }
    text = json.dumps(summary, indent=2, default=str)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(text, encoding="utf-8")
    sys.stdout.write(text + "\n")


if __name__ == "__main__":
    main()
