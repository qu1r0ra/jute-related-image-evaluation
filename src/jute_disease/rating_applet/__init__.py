"""Local blind sitting instrument, under construction."""

import csv
import hashlib
import io
import os
import tempfile
from pathlib import Path
from threading import Lock

from flask import Flask, jsonify, request

from jute_disease.data import audit_row_complete, validate_audit_row


def create_app(sheet: Path) -> Flask:
    """Open a sitting without changing its ratings."""
    sheet = Path(sheet)
    lock = Lock()
    app = Flask(__name__)

    def read() -> tuple[bytes, list[str], list[dict[str, str]]]:
        content = sheet.read_bytes()
        reader = csv.DictReader(io.StringIO(content.decode("utf-8"), newline=""))
        rows = list(reader)
        for row in rows:
            validate_audit_row(row)
        return content, list(reader.fieldnames or []), rows

    @app.get("/api/sitting")
    def sitting() -> object:
        content, _, rows = read()
        complete = [audit_row_complete(row) for row in rows]
        position = next((i + 1 for i, rated in enumerate(complete) if not rated), 1)
        return jsonify(
            position=position,
            completed=sum(complete),
            total=len(rows),
            revision=hashlib.sha256(content).hexdigest(),
        )

    @app.post("/api/items/<int:position>")
    def save(position: int) -> object:
        data = request.get_json()
        with lock:
            content, columns, rows = read()
            if data.get("revision") != hashlib.sha256(content).hexdigest():
                return jsonify(error="The sitting changed. Reload it."), 409
            if not 1 <= position <= len(rows):
                return jsonify(error="Unknown position."), 404
            row = rows[position - 1]
            allowed = {"mask": {"mask_quality", "scene_tag"}, "scene": {"scene_tag"}}
            field = data.get("field")
            if field not in allowed.get(row["kind"], set()):
                return jsonify(error="This choice is unavailable."), 400
            if data.get("value") == row[field]:
                return jsonify(revision=hashlib.sha256(content).hexdigest())
            row[field] = data.get("value")
            try:
                validate_audit_row(row)
            except ValueError, TypeError:
                return jsonify(error="Invalid choice."), 400
            output = io.StringIO(newline="")
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            temp_name = ""
            try:
                with tempfile.NamedTemporaryFile(
                    dir=sheet.parent, delete=False
                ) as temp:
                    temp_name = temp.name
                    temp.write(output.getvalue().encode("utf-8"))
                    temp.flush()
                    os.fsync(temp.fileno())
                os.replace(temp_name, sheet)
            finally:
                if temp_name and os.path.exists(temp_name):
                    os.unlink(temp_name)
        return jsonify(revision=hashlib.sha256(sheet.read_bytes()).hexdigest())

    return app
