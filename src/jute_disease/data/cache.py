"""Bounded lossless image cache and its timing/disk decision.

The cache holds each image downsized so its shortest side is at most
``spec.cache_shortest_side`` (never upscaled), as lossless PNG named by the source
image id. Because the name is content-addressed and the directory is keyed by the
preprocessing digest, a cache can never be mistaken for another spec's or another
image's. Next to the images sits an index of the SHA-256 of every cached file; a
fit trusts the cache only after the files it reads match that index and the
source images still match their ids, and records the identity those rows make.

Every file is published by writing a temporary file unique to its writer and
replacing the target in one step, so concurrent builders and readers never see a
partial file.

This is a decision probe, not a training step: it reads train and validation
images only, stops at a time budget, and records measured time and disk use so
the lazy-decode vs cached-input choice can be made on evidence.
"""

import hashlib
import io
import json
import os
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from jute_disease.data.manifest import Manifest, ManifestEntry
from jute_disease.data.transforms import PreprocessingSpec

REPORT_FILE = "cache_report.json"
DECISION_FILE = "cache_decision.json"
INDEX_FILE = "cache_index.json"
INDEX_VERSION = 1
REPLACE_ATTEMPTS = 8
DEFAULT_BUDGET_SECONDS = 15 * 60
DEFAULT_MAX_BYTES = 20 * 1024**3
PROBE_PARTITIONS = ("train", "val")


class ImageCacheError(ValueError):
    """The image cache is missing, stale, altered or incomplete for a request."""


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
    index_sha256: str = ""

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


def _temporary(target: Path) -> Path:
    """A temporary name no other builder, process or thread can also choose."""
    return target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")


def _retrying[T](action: Callable[[], T]) -> T:
    """Run ``action``, retrying a transient lock.

    Windows refuses to open or replace a file while another process is replacing
    it, which is transient between concurrent builders and readers.
    """
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            return action()
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(0.05 * (attempt + 1))
    raise AssertionError("unreachable")


def _publish_file(temporary: Path, target: Path) -> None:
    """Replace ``target`` with ``temporary`` in one step."""
    _retrying(lambda: os.replace(temporary, target))


def _read_bytes(path: Path) -> bytes:
    return _retrying(path.read_bytes)


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = _temporary(path)
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        _publish_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: object) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _store(
    entry: ManifestEntry, image_root: Path, root: Path, side: int
) -> tuple[int, int, str]:
    """Cache one image, returning its source size, cache size and cache SHA-256.

    The source is hashed first: an image that no longer matches the id the
    manifest froze is refused, never cached under that id.
    """
    source = Path(image_root) / entry.path
    data = source.read_bytes()
    observed = _sha256_bytes(data)
    if observed != entry.image_id:
        raise ImageCacheError(
            f"Source image {entry.path!r} changed after it was frozen: "
            f"expected {entry.image_id}, found {observed}"
        )
    target = cached_path(root, entry.image_id)
    if not target.is_file():
        with Image.open(io.BytesIO(data)) as image:
            small = downsize(image.convert("RGB"), side)
        temporary = _temporary(target)
        try:
            small.save(temporary, format="PNG")
            try:
                _publish_file(temporary, target)
            except PermissionError:
                # A concurrent builder published the same content-named file.
                if not target.is_file():
                    raise
        finally:
            temporary.unlink(missing_ok=True)
    cached = _read_bytes(target)
    return len(data), len(cached), _sha256_bytes(cached)


_INDEX_MEMO: dict[Path, tuple[tuple[int, int], dict[str, object]]] = {}


def _load_index_file(path: Path) -> dict[str, object]:
    """Parse the index at ``path``, reusing the parse while the file is unchanged."""
    stat = _retrying(path.stat)
    stamp = (stat.st_mtime_ns, stat.st_size)
    cached = _INDEX_MEMO.get(path)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    raw = json.loads(_read_bytes(path).decode("utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("image cache index is not an object")
    _INDEX_MEMO[path] = (stamp, raw)
    return raw


def _read_index(root: Path, spec: PreprocessingSpec) -> dict[str, dict[str, object]]:
    """Return the index rows of ``root``, or none if there is no usable index."""
    path = Path(root) / INDEX_FILE
    try:
        raw = _load_index_file(path)
    except OSError, ValueError:
        return {}
    if raw.get("version") != INDEX_VERSION or raw.get("spec_digest") != spec.digest():
        return {}
    images = raw.get("images")
    return images if isinstance(images, dict) else {}


def build_image_cache(
    manifest: Manifest,
    image_root: Path,
    cache_dir: Path,
    spec: PreprocessingSpec,
    partitions: tuple[str, ...] = PROBE_PARTITIONS,
    budget_seconds: float = DEFAULT_BUDGET_SECONDS,
    workers: int = 4,
    allow_test: bool = False,
    clock: Callable[[], float] = time.perf_counter,
) -> CacheReport:
    """Build the cache for the chosen partitions within a time budget.

    Raises:
        ValueError: the test partition is requested without ``allow_test``.
    """
    if "test" in partitions and not allow_test:
        raise ValueError("The test partition is not read by the cache probe")
    entries = [e for name in partitions for e in manifest.partition(name)]
    root = cache_root(cache_dir, spec)
    root.mkdir(parents=True, exist_ok=True)
    start = clock()
    done = source_bytes = cache_bytes = 0
    rows: dict[str, dict[str, object]] = {}
    chunk = max(1, workers * 4)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for begin in range(0, len(entries), chunk):
            if clock() - start > budget_seconds:
                break
            batch = entries[begin : begin + chunk]
            results = pool.map(
                lambda e: _store(e, image_root, root, spec.cache_shortest_side), batch
            )
            for entry, (source_size, cache_size, cache_sha) in zip(
                batch, results, strict=True
            ):
                done += 1
                source_bytes += source_size
                cache_bytes += cache_size
                rows[entry.image_id] = {"sha256": cache_sha, "bytes": cache_size}
    index_sha = _publish_index(root, spec, rows)
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
        index_sha256=index_sha,
    )
    _write_json_atomic(root / REPORT_FILE, report.to_json())
    return report


def _publish_index(
    root: Path, spec: PreprocessingSpec, rows: dict[str, dict[str, object]]
) -> str:
    """Merge ``rows`` into the index of ``root`` and publish it atomically.

    Rows are only ever added or refreshed, so builders of different manifests
    share one cache; builders of the same manifest write identical bytes.
    """
    merged = {**_read_index(root, spec), **rows}
    text = (
        json.dumps(
            {
                "version": INDEX_VERSION,
                "spec_digest": spec.digest(),
                "images": merged,
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    _write_text_atomic(root / INDEX_FILE, text)
    return _sha256_bytes(text.encode("utf-8"))


def _index_rows(
    cache_dir: Path,
    spec: PreprocessingSpec,
    manifest: Manifest,
    partitions: tuple[str, ...],
) -> list[tuple[ManifestEntry, str]]:
    root = cache_root(cache_dir, spec)
    index = _read_index(root, spec)
    if not index:
        raise ImageCacheError(
            f"No image cache index for spec {spec.digest()[:16]} under {cache_dir}; "
            "build it with `freeze cache`"
        )
    rows: list[tuple[ManifestEntry, str]] = []
    for name in partitions:
        for entry in manifest.partition(name):
            row = index.get(entry.image_id)
            if row is None:
                raise ImageCacheError(
                    f"Image {entry.path!r} ({entry.image_id[:12]}) is not in the "
                    f"image cache under {cache_dir}"
                )
            rows.append((entry, str(row["sha256"])))
    return rows


def _rows_identity(spec: PreprocessingSpec, rows: list[tuple[str, str]]) -> str:
    payload = json.dumps([spec.digest(), sorted(rows)], separators=(",", ":"))
    return _sha256_bytes(payload.encode("utf-8"))


def image_cache_identity(
    cache_dir: Path,
    spec: PreprocessingSpec,
    manifest: Manifest,
    partitions: tuple[str, ...],
) -> str:
    """The identity of the cache rows these partitions read, from the index alone.

    It is the SHA-256 of the spec digest and the index SHA-256 of every image, so
    it changes whenever the preprocessing or any cached image does.

    Raises:
        ImageCacheError: there is no index for the spec or an image is missing.
    """
    rows = _index_rows(cache_dir, spec, manifest, partitions)
    return _rows_identity(spec, [(entry.image_id, sha) for entry, sha in rows])


def verify_image_cache(
    cache_dir: Path,
    spec: PreprocessingSpec,
    manifest: Manifest,
    partitions: tuple[str, ...],
    image_root: Path,
) -> str:
    """Check the cache against its index and the sources, and return its identity.

    Every cached file a fit would read must match the SHA-256 the index recorded
    for it, and every source image must still hash to the id the manifest froze,
    so a cache built before an image changed is refused rather than read.

    Raises:
        ImageCacheError: the index is missing, a row is missing or altered, or a
            source image changed after the cache was built.
    """
    rows = _index_rows(cache_dir, spec, manifest, partitions)
    root = cache_root(cache_dir, spec)
    for entry, expected in rows:
        source = Path(image_root) / entry.path
        try:
            observed_source = _sha256_bytes(source.read_bytes())
        except OSError as error:
            raise ImageCacheError(
                f"Source image {entry.path!r} cannot be read: {error}"
            ) from error
        if observed_source != entry.image_id:
            raise ImageCacheError(
                f"Source image {entry.path!r} changed after the cache was built: "
                f"expected {entry.image_id}, found {observed_source}"
            )
        try:
            observed = _sha256_bytes(_read_bytes(cached_path(root, entry.image_id)))
        except OSError as error:
            raise ImageCacheError(
                f"Cached image {entry.image_id[:12]} cannot be read: {error}"
            ) from error
        if observed != expected:
            raise ImageCacheError(
                f"Cached image {entry.image_id[:12]} does not match the cache index"
            )
    return _rows_identity(spec, [(entry.image_id, sha) for entry, sha in rows])


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
