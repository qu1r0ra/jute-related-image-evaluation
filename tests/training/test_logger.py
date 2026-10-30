"""Logging stays local-first: W&B is optional and never reaches the network."""

import pytest

from jute_disease.training import resolve_wandb_mode


def test_no_credentials_means_offline_not_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    assert resolve_wandb_mode(None) == "offline"


def test_explicit_off_disables_wandb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "x" * 40)
    assert resolve_wandb_mode("off") == "off"


def test_online_needs_a_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert resolve_wandb_mode("online") == "offline"
    monkeypatch.setenv("WANDB_API_KEY", "x" * 40)
    assert resolve_wandb_mode("online") == "online"


def test_environment_disabled_mode_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "x" * 40)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    assert resolve_wandb_mode("online") == "off"


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="wandb mode"):
        resolve_wandb_mode("sometimes")


def test_no_package_module_imports_wandb_at_import_time() -> None:
    # timm imports wandb itself when installed, so check our own sources.
    import ast

    from jute_disease.config import ROOT_DIR

    offenders = []
    for source in (ROOT_DIR / "src" / "jute_disease").rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in tree.body:
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


class _FakeWandb:
    created: list[dict] = []
    broken = False

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        type(self).created.append(kwargs)

    @property
    def experiment(self) -> object:
        if type(self).broken:
            raise RuntimeError("wandb-core did not start")
        return object()


@pytest.fixture
def fake_wandb(monkeypatch: pytest.MonkeyPatch) -> type[_FakeWandb]:
    import lightning.pytorch.loggers as loggers

    _FakeWandb.created = []
    _FakeWandb.broken = False
    monkeypatch.setattr(loggers, "WandbLogger", _FakeWandb)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    return _FakeWandb


def test_metrics_land_in_one_fixed_layout(tmp_path, fake_wandb) -> None:
    from jute_disease.training import experiment_loggers

    loggers, status = experiment_loggers(tmp_path, wandb_mode="off")
    assert status == "off"
    assert len(loggers) == 1
    assert fake_wandb.created == []
    assert loggers[0].log_dir == str(tmp_path / "metrics" / "version_0")


def test_requested_online_without_a_key_is_created_offline(
    tmp_path, fake_wandb
) -> None:
    from jute_disease.training import experiment_loggers

    loggers, status = experiment_loggers(tmp_path, wandb_mode="online")
    assert status == "offline"
    assert len(loggers) == 2
    assert fake_wandb.created[0]["offline"] is True


def test_a_wandb_service_failure_never_fails_the_run(tmp_path, fake_wandb) -> None:
    from jute_disease.training import experiment_loggers

    fake_wandb.broken = True
    loggers, status = experiment_loggers(tmp_path, wandb_mode="offline")
    assert status == "unavailable"
    assert len(loggers) == 1
