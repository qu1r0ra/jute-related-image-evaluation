"""Local blind sitting instrument and its private persistence journal."""

import csv
import hashlib
import io
import json
import os
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from PIL import Image, ImageOps

from jute_disease.data import MASK_RUBRIC, audit_row_complete, validate_audit_row

_VERSION = "1.3.0"
_PREVIEW_SIZE = (1400, 1400)
_PAIR_RATINGS = ("exact", "near-duplicate", "very similar", "different")
_PAIR_BOUNDARY = {
    "near_duplicate": (
        "the same physical leaf or scene captured again, such as a burst frame, "
        "a small shift or zoom, a crop or a re-save"
    ),
    "very_similar": "a different leaf or scene that merely looks alike",
    "merge": (
        "Exact and near-duplicate mean merge; very similar and different mean no merge."
    ),
}
_LOCKS: dict[str, Lock] = {}
_LOCKS_GUARD = Lock()


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _runtime(sheet: Path) -> Path:
    return sheet.parent / ".rating-applet" / sheet.name


@contextmanager
def _locked(runtime: Path) -> Iterator[None]:
    runtime.mkdir(parents=True, exist_ok=True)
    with _LOCKS_GUARD:
        local = _LOCKS.setdefault(str(runtime.resolve()), Lock())
    with local, (runtime / "lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            deadline = time.monotonic() + 30
            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _append(runtime: Path, event: dict[str, object]) -> None:
    record = {
        "schema": 1,
        "timestamp": datetime.now(UTC).isoformat(),
        "instrument_version": _VERSION,
        **event,
    }
    with (runtime / "events.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read(sheet: Path) -> tuple[bytes, list[str], list[dict[str, str]]]:
    content = sheet.read_bytes()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8"), newline=""))
    rows = list(reader)
    for row in rows:
        validate_audit_row(row)
    return content, list(reader.fieldnames or []), rows


def _state(runtime: Path, content: bytes) -> tuple[bool, bool]:
    """Reconcile a single interrupted transaction; reject unknown divergence."""
    path = runtime / "events.jsonl"
    if not path.exists():
        return False, False
    raw = path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("Incomplete journal; restore from backup and seek assistance.")
    try:
        events = [json.loads(line) for line in raw.splitlines()]
        expected = None
        pending = None
        finished = False
        for event in events:
            if event["schema"] != 1:
                raise ValueError("Unknown journal version")
            kind = event["type"]
            if kind == "adopt":
                if expected is None:
                    expected = event["digest"]
                elif expected != event["digest"] or pending:
                    raise ValueError("Conflicting adoption")
            elif kind == "prepare":
                if pending or event["before"] != expected:
                    raise ValueError("Conflicting transaction")
                pending = event
            elif kind in ("commit", "abort"):
                if not pending or event["transaction"] != pending["transaction"]:
                    raise ValueError("Orphan transaction")
                if kind == "commit":
                    expected = pending["after"]
                pending = None
            elif kind in ("finish", "reopen"):
                if pending or event["digest"] != expected:
                    raise ValueError("Conflicting lifecycle")
                finished = kind == "finish"
            else:
                raise ValueError("Unknown event")
        actual = _digest(content)
        if pending:
            if actual == pending["before"]:
                outcome = "abort"
            elif actual == pending["after"]:
                outcome = "commit"
                expected = actual
            else:
                raise ValueError("Ambiguous interrupted save")
            _append(
                runtime,
                {
                    "type": outcome,
                    "transaction": pending["transaction"],
                    "recovered": True,
                },
            )
        if expected != actual:
            raise ValueError("Sheet differs from its recorded revision")
        return finished, True
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Invalid journal; restore from backup and seek assistance."
        ) from exc


def _revision(runtime: Path, content: bytes) -> str:
    """Include history so an earlier byte-identical state is still stale."""
    history = hashlib.sha256(content)
    for line in (runtime / "events.jsonl").read_bytes().splitlines():
        if json.loads(line)["type"] in {"commit", "finish", "reopen"}:
            history.update(line)
    return history.hexdigest()


def _collected(sheet: Path) -> bool:
    return any(
        (sheet.parent / name).exists()
        for name in ("owner.csv", "owner-masks.csv", "owner-scenes.csv")
    )


def reopen_sitting(sheet: Path, reason: str) -> None:
    """Record an explicit external correction authorization for a finished sitting."""
    sheet = Path(sheet).resolve()
    if not reason.strip():
        raise ValueError("A correction reason is required.")
    runtime = _runtime(sheet)
    with _locked(runtime):
        content, _, _ = _read(sheet)
        finished, initialized = _state(runtime, content)
        if not initialized or not finished or _collected(sheet):
            raise ValueError("This sitting cannot be reopened.")
        _append(
            runtime, {"type": "reopen", "digest": _digest(content), "reason": reason}
        )


def create_app(sheet: Path, pool_roots: dict[str, Path] | None = None) -> Flask:
    """Open a sitting without changing its ratings."""
    sheet = Path(sheet).resolve()
    runtime = _runtime(sheet)
    session = uuid.uuid4().hex
    resolved_pool_roots = {
        name: Path(path).resolve() for name, path in (pool_roots or {}).items()
    }
    app = Flask(__name__)

    @app.after_request
    def prevent_caching(response: Response) -> Response:
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.before_request
    def local_request() -> object:
        host = request.host.partition(":")[0].lower()
        origin = request.headers.get("Origin")
        top_level_navigation = (
            request.method == "GET"
            and request.path == "/"
            and request.headers.get("Sec-Fetch-Mode") == "navigate"
            and request.headers.get("Sec-Fetch-Dest") == "document"
        )
        if (
            host not in {"127.0.0.1", "localhost"}
            or (
                request.headers.get("Sec-Fetch-Site") == "cross-site"
                and not top_level_navigation
            )
            or (origin is not None and origin != request.host_url.rstrip("/"))
        ):
            return "Forbidden.", 403
        return None

    @app.get("/")
    def index() -> object:
        return send_from_directory(Path(__file__).parent / "static", "index.html")

    with _locked(runtime):
        try:
            content, _, rows = _read(sheet)
            _state(runtime, content)
        except OSError, ValueError, KeyError, csv.Error:
            # Preserve the evidence; requests report the recovery state below.
            pass
        else:
            backup = runtime / f"{session}.backup"
            with backup.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            _append(
                runtime,
                {
                    "type": "adopt",
                    "session": session,
                    "digest": _digest(content),
                    "completed_items": [
                        row["item_id"] for row in rows if audit_row_complete(row)
                    ],
                },
            )

    @app.get("/api/items/<int:position>")
    def item(position: int) -> object:
        try:
            with _locked(runtime):
                content, _, rows = _read(sheet)
                _state(runtime, content)
                if not 1 <= position <= len(rows):
                    return jsonify(error="Unknown position."), 404
                row = rows[position - 1]
                if row["kind"] == "pair":
                    return jsonify(
                        kind="pair",
                        rating=row["rating"],
                        ratings=list(_PAIR_RATINGS),
                        pair_boundary=_PAIR_BOUNDARY,
                    )
                if row["kind"] not in {"mask", "scene"}:
                    return jsonify(error="Unsupported item."), 400
                rubric = json.loads(
                    (sheet.parent / "kit.json").read_text(encoding="utf-8")
                )
                anchors = rubric.get("scene_rubric", {}).get("anchors", {})
                return jsonify(
                    kind=row["kind"],
                    mask_quality=row["mask_quality"],
                    scene_tag=row["scene_tag"],
                    mask_anchors=MASK_RUBRIC["anchors"],
                    scene_anchors={
                        key: anchors.get(key, "")
                        for key in ("clean-background", "field", "unknown")
                    },
                )
        except OSError, ValueError, KeyError, csv.Error, TypeError:
            return jsonify(error="Unable to load this item."), 409

    @app.get("/images/<int:position>/<variant>", defaults={"side": None})
    @app.get("/images/<int:position>/<side>/<variant>")
    def image(position: int, variant: str, side: str | None) -> object:
        if variant not in {"preview", "original"}:
            return "Image unavailable.", 404
        try:
            with _locked(runtime):
                content, _, rows = _read(sheet)
                _state(runtime, content)
                if not 1 <= position <= len(rows):
                    raise ValueError("Unknown position")
                row = rows[position - 1]
                if row["kind"] == "pair":
                    if side not in {"a", "b"}:
                        raise ValueError("Unknown side")
                    field = "image_a" if side == "a" else "image_b"
                    kit = json.loads(
                        (sheet.parent / "kit.json").read_text(encoding="utf-8")
                    )
                    item = next(
                        item
                        for item in kit["items"]
                        if item["item_id"] == row["item_id"]
                    )
                    pool = item.get("pool")
                    root = resolved_pool_roots[pool]
                    image_path = Path(row[field])
                    if image_path.parts and image_path.parts[0] == pool:
                        image_path = Path(*image_path.parts[1:])
                else:
                    if side is not None:
                        raise ValueError("Pair side is only available for pair items")
                    field = {"mask": "overlay_path", "scene": "image_a"}.get(
                        row["kind"]
                    )
                    if field is None:
                        raise ValueError("Unsupported kind")
                    root = sheet.parent.resolve()
                    image_path = Path(row[field])
                path = (root / image_path).resolve(strict=True)
                if not path.is_relative_to(root) or not path.is_file():
                    raise ValueError("Image outside kit")
                source_bytes = path.read_bytes()
                cache_version = f"{_VERSION}:{variant}"
                if variant == "preview":
                    cache_version += f":{_PREVIEW_SIZE}"
                cache_key = _digest(f"{cache_version}:".encode() + source_bytes)
                cache = runtime / f"{variant}s" / f"{cache_key}.png"
                if not cache.is_file():
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    with Image.open(io.BytesIO(source_bytes)) as source:
                        picture = ImageOps.exif_transpose(source).convert("RGB")
                        if variant == "preview":
                            picture.thumbnail(_PREVIEW_SIZE, Image.Resampling.LANCZOS)
                        temp_name = ""
                        try:
                            with tempfile.NamedTemporaryFile(
                                dir=cache.parent, delete=False, suffix=".tmp"
                            ) as temp:
                                temp_name = temp.name
                                picture.save(temp, format="PNG")
                                temp.flush()
                                os.fsync(temp.fileno())
                            os.replace(temp_name, cache)
                        finally:
                            if temp_name and os.path.exists(temp_name):
                                os.unlink(temp_name)
                response = send_file(cache, mimetype="image/png", max_age=0)
                response.headers["Cache-Control"] = "no-store"
                return response
        except OSError, ValueError, KeyError, csv.Error, Image.DecompressionBombError:
            return "Image unavailable.", 404

    @app.get("/api/sitting")
    def sitting() -> object:
        try:
            with _locked(runtime):
                content, _, rows = _read(sheet)
                finished, initialized = _state(runtime, content)
                if not initialized:
                    raise ValueError("No adoption record")
                complete = [audit_row_complete(row) for row in rows]
                position = next(
                    (i + 1 for i, rated in enumerate(complete) if not rated), 1
                )
                return jsonify(
                    position=position,
                    completed=sum(complete),
                    total=len(rows),
                    revision=_revision(runtime, content),
                    read_only=finished or _collected(sheet),
                    finished=finished,
                )
        except OSError, ValueError, KeyError, csv.Error:
            return jsonify(
                error=(
                    "Recovery required. Preserve the sheet and private backup; "
                    "inspect the journal before continuing."
                )
            ), 409

    @app.post("/api/items/<int:position>")
    def save(position: int) -> object:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(error="Invalid request."), 400
        try:
            with _locked(runtime):
                content, columns, rows = _read(sheet)
                finished, initialized = _state(runtime, content)
                if not initialized:
                    raise ValueError("No adoption record")
                if finished or _collected(sheet):
                    return jsonify(error="This sitting is read-only."), 403
                if data.get("revision") != _revision(runtime, content):
                    return jsonify(error="The sitting changed. Reload it."), 409
                if not 1 <= position <= len(rows):
                    return jsonify(error="Unknown position."), 404
                row = rows[position - 1]
                allowed = {
                    "mask": {"mask_quality", "scene_tag"},
                    "scene": {"scene_tag"},
                    "pair": {"rating"},
                }
                field = data.get("field")
                if (
                    not isinstance(field, str)
                    or field not in allowed.get(row["kind"], set())
                    or not isinstance(data.get("value"), str)
                ):
                    return jsonify(error="Invalid choice."), 400
                previous = row[field]
                if data["value"] == previous:
                    return jsonify(revision=_revision(runtime, content))
                row[field] = data["value"]
                try:
                    validate_audit_row(row)
                except ValueError, TypeError:
                    return jsonify(error="Invalid choice."), 400
                output = io.StringIO(newline="")
                writer = csv.DictWriter(output, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
                updated = output.getvalue().encode("utf-8")
                transaction = uuid.uuid4().hex
                temp_name = ""
                try:
                    with tempfile.NamedTemporaryFile(
                        dir=sheet.parent, delete=False
                    ) as temp:
                        temp_name = temp.name
                        temp.write(updated)
                        temp.flush()
                        os.fsync(temp.fileno())
                    _append(
                        runtime,
                        {
                            "type": "prepare",
                            "transaction": transaction,
                            "session": session,
                            "item_id": row["item_id"],
                            "field": field,
                            "previous": previous,
                            "new": data["value"],
                            "before": _digest(content),
                            "after": _digest(updated),
                        },
                    )
                    if sheet.read_bytes() != content:
                        raise ValueError("Sheet changed during save")
                    os.replace(temp_name, sheet)
                    if os.name != "nt":
                        directory = os.open(sheet.parent, os.O_RDONLY)
                        try:
                            os.fsync(directory)
                        finally:
                            os.close(directory)
                    _append(runtime, {"type": "commit", "transaction": transaction})
                finally:
                    if temp_name and os.path.exists(temp_name):
                        os.unlink(temp_name)
                return jsonify(revision=_revision(runtime, updated))
        except OSError, ValueError, KeyError, csv.Error:
            return jsonify(
                error=(
                    "Save failed or recovery is required. Reload and inspect "
                    "the private backup if the problem persists."
                )
            ), 409

    @app.post("/api/finish")
    def finish() -> object:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(error="Invalid request."), 400
        try:
            with _locked(runtime):
                content, _, rows = _read(sheet)
                finished, initialized = _state(runtime, content)
                if not initialized:
                    raise ValueError("No adoption record")
                if finished or _collected(sheet):
                    return jsonify(error="This sitting is read-only."), 403
                if (
                    data.get("revision") != _revision(runtime, content)
                    or not rows
                    or any(row["kind"] not in {"mask", "scene", "pair"} for row in rows)
                    or not all(map(audit_row_complete, rows))
                ):
                    return jsonify(
                        error="Reload or complete the sitting before finishing."
                    ), 409
                _append(
                    runtime,
                    {"type": "finish", "digest": _digest(content), "session": session},
                )
                return jsonify(read_only=True, revision=_revision(runtime, content))
        except OSError, ValueError, KeyError, csv.Error:
            return jsonify(error="Finish failed; recovery may be required."), 409

    return app
