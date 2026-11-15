"""Commands that do not train or load models start without the training stack."""

import json
import subprocess
import sys

import pytest

HEAVY_MODULES = ("torch", "lightning", "timm", "torchvision", "torchmetrics")

PROBE = """
import json
import sys

from jute_disease.cli import main

try:
    main(json.loads(sys.argv[1]))
except SystemExit:
    pass
print(json.dumps(sorted(m for m in {heavy!r} if m in sys.modules)))
"""


def _loaded_heavy_modules(arguments: list[str]) -> list[str]:
    """Run the CLI in a fresh interpreter and list the heavy modules it imported."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            PROBE.format(heavy=HEAVY_MODULES),
            json.dumps(arguments),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--help"],
        ["queue", "plan"],
        ["freeze", "--help"],
        ["data", "--help"],
        ["audit", "--help"],
        ["evaluate", "--help"],
    ],
    ids=lambda arguments: " ".join(arguments) or "import",
)
def test_commands_that_do_not_train_skip_the_training_stack(
    arguments: list[str],
) -> None:
    assert _loaded_heavy_modules(arguments) == []
