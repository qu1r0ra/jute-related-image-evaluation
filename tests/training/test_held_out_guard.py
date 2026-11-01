"""Static and command-line guards: training has no route to the test partition.

The data layer refuses the partition at run time; these guards catch a new
training module or command that would try to ask for it in the first place.
"""

import ast
from pathlib import Path

import pytest

from jute_disease import training
from jute_disease.cli import build_parser

TRAINING_PACKAGE = Path(training.__file__).parent
HELD_OUT = {"test", "predict"}


def _is_exclusion(node: ast.Constant, parents: dict[ast.AST, ast.AST]) -> bool:
    """A name passed to ``str.startswith`` filters the partition out, not in."""
    call = parents.get(node)
    return (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "startswith"
    )


def test_no_training_module_names_the_held_out_partition() -> None:
    offenders: list[str] = []
    for path in sorted(TRAINING_PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and node.value in HELD_OUT
                and not _is_exclusion(node, parents)
            ):
                offenders.append(f"{path.relative_to(TRAINING_PACKAGE)}:{node.lineno}")
    assert offenders == []


@pytest.mark.parametrize("action", ["test", "predict", "validate"])
def test_train_offers_no_evaluation_subcommand(action: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["train", action])
    assert excinfo.value.code == 2


@pytest.mark.parametrize("option", ["--allow-test", "--partition=test"])
def test_train_fit_has_no_option_that_opens_the_test_partition(option: str) -> None:
    fit = ["train", "fit", "--workspace", "w", "--fit-id", "f"]
    fit += ["--attempt-dir", "a", "--result", "r.json"]
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args([*fit, option])
    assert excinfo.value.code == 2
