"""Local blind sitting instrument."""

import csv
from pathlib import Path

from flask import Flask, jsonify

from jute_disease.data import audit_row_complete, validate_audit_row


def create_app(sheet: Path) -> Flask:
    """Open a sitting without changing its ratings."""
    with Path(sheet).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        validate_audit_row(row)
    app = Flask(__name__)

    @app.get("/api/sitting")
    def sitting() -> object:
        complete = [audit_row_complete(row) for row in rows]
        position = next((i + 1 for i, rated in enumerate(complete) if not rated), 1)
        return jsonify(position=position, completed=sum(complete), total=len(rows))

    return app
