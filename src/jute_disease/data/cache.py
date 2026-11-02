"""Bounded lossless image cache and its timing/disk decision.

The cache holds each image downsized so its shortest side is at most
``spec.cache_shortest_side`` (never upscaled), as lossless PNG named by the source
image id. Because the name is content-addressed and the directory is keyed by the
preprocessing digest, a cache can never be mistaken for another spec's or another
image's. A cache is trusted only when its report says it is complete for the
manifest it was built from.

This is a decision probe, not a training step: it reads train and validation
images only, stops at a time budget, and records measured time and disk use so
the lazy-decode vs cached-input choice can be made on evidence.
"""

import json
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from jute_disease.data.manifest import HeldOutPartitionError, Manifest, ManifestEntry
from jute_disease.data.transforms import PreprocessingSpec

REPORT_FILE = "cache_report.json"
DECISION_FILE = "cache_decision.json"
DEFAULT_BUDGET_SECONDS = 15 * 60
DEFAULT_MAX_BYTES = 20 * 1024**3
PROBE_PARTITIONS = ("train", "val")


@dataclass(frozen=True)
class CacheReport:
    """Measured outcome of one cache build."""

    spec_digest: str
    manifest_digest: str
    partitions: tuple[str, ...]
    images_total: int
    images_done: int
    source_bytes: int
    cache_bytes: int
    seconds: float
    budget_seconds: float
    complete: bool

    def to_json(self) -> dict:
        """Return a JSON-safe mapping."""
        return {**asdict(self), "partitions": list(self.partitions)}


def cache_root(cache_dir: Path, spec: PreprocessingSpec) -> Path:
    """Return the directory holding the cache for ``spec``."""
    return Path(cache_dir) / spec.digest()[:16]


def cached_path(root: Path, image_id: str) -> Path:
    """Return the cache file for one source image id."""
    return Path(root) / f"{image_id}.png"


def downsize(image: Image.Image, shortest_side: int) -> Image.Image:
    """Downsize so the shortest side is at most ``shortest_side``; never upscale."""
    width, height = image.size
    shortest = min(width, height)
    if shortest <= shortest_side:
        return image
    scale = shortest_side / shortest
    return image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.BICUBIC,
    )


def _store(
    entry: ManifestEntry, image_root: Path, root: Path, side: int
) -> tuple[int, int]:
    target = cached_path(root, entry.image_id)
    source = Path(image_root) / entry.path
    if target.is_file():
        return source.stat().st_size, target.stat().st_size
    with Image.open(source) as image:
        small = downsize(image.convert("RGB"), side)
    temporary = target.with_name(target.name + ".tmp")
    small.save(temporary, format="PNG")
    os.replace(temporary, target)
    return source.stat().st_size, target.stat().st_size


def build_image_cache(
    manifest: Manifest,
    image_root: Path,
    cache_dir: Path,
    spec: PreprocessingSpec,
    partitions: tuple[str, ...] = PROBE_PARTITIONS,
    budget_seconds: float = DEFAULT_BUDGET_SECONDS,
    workers: int = 4,
    clock: Callable[[], float] = time.perf_counter,
) -> CacheReport:
    """Build the cache for the chosen partitions within a time budget.

    Raises:
        HeldOutPartitionError: the test partition is requested.
    """
    if "test" in partitions:
        raise HeldOutPartitionError(
            "The test partition is only readable by the gated evaluator"
        )
    entries = [e for name in partitions for e in manifest.partition(name)]
    root = cache_root(cache_dir, spec)
    root.mkdir(parents=True, exist_ok=True)
    (root / REPORT_FILE).unlink(missing_ok=True)
    start = clock()
    done = source_bytes = cache_bytes = 0
    chunk = max(1, workers * 4)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for begin in range(0, len(entries), chunk):
            if clock() - start > budget_seconds:
                break
            batch = entries[begin : begin + chunk]
            results = pool.map(
                lambda e: _store(e, image_root, root, spec.cache_shortest_side), batch
            )
            for source_size, cache_size in results:
                done += 1
                source_bytes += source_size
                cache_bytes += cache_size
    report = CacheReport(
        spec_digest=spec.digest(),
        manifest_digest=manifest.digest(),
        partitions=tuple(partitions),
        images_total=len(entries),
        images_done=done,
        source_bytes=source_bytes,
        cache_bytes=cache_bytes,
        seconds=clock() - start,
        budget_seconds=budget_seconds,
        complete=done == len(entries),
    )
    (root / REPORT_FILE).write_text(
        json.dumps(report.to_json(), indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return report


def is_cache_complete(
    root: Path, manifest: Manifest, partitions: tuple[str, ...]
) -> bool:
    """Return whether ``root`` holds a completed cache for these partitions."""
    report_path = Path(root) / REPORT_FILE
    if not report_path.is_file():
        return False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return (
        report["complete"]
        and report["manifest_digest"] == manifest.digest()
        and set(partitions) <= set(report["partitions"])
    )


def decide_cache(
    report: CacheReport, max_bytes: int = DEFAULT_MAX_BYTES
) -> dict[str, object]:
    """Decide between cached inputs and lazy decoding from measured evidence.

    The decision uses only elapsed time and disk use, never model behaviour.
    """
    reasons = []
    if not report.complete:
        reasons.append(
            f"build did not finish within {report.budget_seconds:.0f}s "
            f"({report.images_done}/{report.images_total} images)"
        )
    if report.cache_bytes > max_bytes:
        reasons.append(f"cache needs {report.cache_bytes} bytes (limit {max_bytes})")
    return {
        "decision": "adopt-cache" if not reasons else "lazy-decode",
        "reasons": reasons or ["completed within budget and disk limit"],
        "evidence": report.to_json(),
        "limits": {"budget_seconds": report.budget_seconds, "max_bytes": max_bytes},
    }


def write_decision(decision: dict[str, object], root: Path) -> Path:
    """Persist a decision next to the cache it describes."""
    path = Path(root) / DECISION_FILE
    path.write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return path
