"""Frozen synthetic target and a stand-in repository tree for the evaluator tests."""

import json
from pathlib import Path

import pytest

from jute_disease.config import CURATION_RULES_PATH
from jute_disease.data import freeze_target, make_synthetic_data


@pytest.fixture(scope="module")
def target(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base = tmp_path_factory.mktemp("gated")
    plan = make_synthetic_data(base / "images", per_class=20, size=64)
    rules = json.loads(CURATION_RULES_PATH.read_text(encoding="utf-8"))
    rules["min_support"] = 2
    (base / "rules.json").write_text(json.dumps(rules), encoding="utf-8")
    freeze_target(
        base / "freeze",
        image_root=plan.root,
        rules_path=base / "rules.json",
        reviews_path=None,
        workers=2,
    )
    return base / "freeze" / "target"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "repo" / "src").mkdir(parents=True)
    (tmp_path / "repo" / "configs").mkdir()
    (tmp_path / "repo" / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "repo" / "configs" / "c.json").write_text("{}", encoding="utf-8")
    (tmp_path / "repo" / "uv.lock").write_text("lock", encoding="utf-8")
    return tmp_path / "repo"
