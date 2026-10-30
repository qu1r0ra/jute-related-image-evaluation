import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jute_disease.config import CHECKPOINTS_DIR
from jute_disease.training.grid_search import run_grid_search


@pytest.fixture
def phase_1_grid(tmp_path: Path) -> Path:
    config_content = """
transfer_learning_levels:
  - name: "level_1_imagenet"
    weights_path: "imagenet"

dropout_rates:
  - 0.1

fixed_params:
  input_resolution: [256, 256]
  batch_size: 32
  max_epochs: 30
    """
    path = tmp_path / "phase1_grid.yaml"
    path.write_text(config_content)
    return path


@pytest.fixture
def phase_2_grid(tmp_path: Path) -> Path:
    config_content = f"""
learning_rate:
  - 0.005

weight_decay:
  - 0.1

locked_params:
  name: "level_3_plantdoc"
  weights_path: >-
    {CHECKPOINTS_DIR / "pretrained" / "mobilenet_v2-plantvillage-plantdoc.ckpt"}
  dropout_rate: 0.3

fixed_params:
  input_resolution: [256, 256]
  batch_size: 32
  max_epochs: 50
    """
    path = tmp_path / "phase2_grid.yaml"
    path.write_text(config_content)
    return path


def test_run_grid_search_phase1(
    phase_1_grid: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mock_run = MagicMock()
    monkeypatch.setattr(subprocess, "run", mock_run)
    mock_agg = MagicMock()
    monkeypatch.setattr(
        "jute_disease.training.grid_search._aggregate_metrics", mock_agg
    )

    base_config = tmp_path / "mobilenet_v2.yaml"
    base_config.touch()

    run_grid_search(phase_1_grid, base_config)

    assert mock_run.call_count == 1
    fit_cmd = mock_run.call_args_list[0][0][0]

    # Check that appropriate phase 1 args are passed
    assert "jute_disease.cli" in fit_cmd
    assert "fit" in fit_cmd
    assert "--config" in fit_cmd
    assert any("drop_rate=0.1" in c for c in fit_cmd)
    assert any("pretrained=True" in c for c in fit_cmd)
    assert any("checkpoint_path=null" in c for c in fit_cmd)

    # The held-out partition is never scored here
    assert "test" not in fit_cmd

    # Check aggregation
    assert mock_agg.call_count == 1
    assert "mobilenet_v2-l1_imagenet-dr_0.1" in mock_agg.call_args[0][0]


def test_run_grid_search_phase2(
    phase_2_grid: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mock_run = MagicMock()
    monkeypatch.setattr(subprocess, "run", mock_run)
    mock_agg = MagicMock()
    monkeypatch.setattr(
        "jute_disease.training.grid_search._aggregate_metrics", mock_agg
    )

    base_config = tmp_path / "mobilenet_v2.yaml"
    base_config.touch()

    run_grid_search(phase_2_grid, base_config)

    assert mock_run.call_count == 1
    fit_cmd = mock_run.call_args_list[0][0][0]

    # Check that appropriate phase 2 args are passed
    assert "jute_disease.cli" in fit_cmd
    assert "fit" in fit_cmd
    assert "--config" in fit_cmd
    assert any("drop_rate=0.3" in c for c in fit_cmd)
    assert any("pretrained=False" in c for c in fit_cmd)
    ckpt_str = "--model.feature_extractor.init_args.checkpoint_path="
    expected = f"{ckpt_str}{CHECKPOINTS_DIR / 'pretrained' / 'mobilenet_v2-plantvillage-plantdoc.ckpt'}"
    assert any(expected in c for c in fit_cmd)
    assert any("lr=0.005" in c for c in fit_cmd)
    assert any("weight_decay=0.1" in c for c in fit_cmd)

    # The held-out partition is never scored here
    assert "test" not in fit_cmd

    # Check aggregation
    assert mock_agg.call_count == 1
    assert "mobilenet_v2-l3_plantdoc-lr_0.005-wd_0.1" in mock_agg.call_args[0][0]
