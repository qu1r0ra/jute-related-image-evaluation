"""Logging stays local: metrics go to CSV and no package module imports W&B."""

from pathlib import Path


def test_no_package_module_imports_wandb() -> None:
    import ast

    from jute_disease.config import ROOT_DIR

    offenders = []
    for source in (ROOT_DIR / "src" / "jute_disease").rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            if any(name.split(".")[0] == "wandb" for name in names):
                offenders.append(source.name)
    assert offenders == []


def test_metrics_land_in_one_fixed_layout(tmp_path: Path) -> None:
    from jute_disease.training import experiment_loggers

    loggers = experiment_loggers(tmp_path)
    assert len(loggers) == 1
    assert loggers[0].log_dir == str(tmp_path / "metrics" / "version_0")
