"""What a finished fit hands back to the queue."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FitResult:
    """Receipt results plus the files whose bytes the receipt must pin."""

    results: dict[str, Any]
    artifacts: dict[str, Path]
