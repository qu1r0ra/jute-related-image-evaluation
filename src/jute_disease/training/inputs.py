"""Where the frozen target and verified source packages live on disk."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataInputs:
    """Frozen manifests and image roots a fit may read.

    ``sources`` maps a source stage name to its manifest directory and the pinned
    package root its images live under.
    """

    image_root: Path
    target_dir: Path
    sources: Mapping[str, tuple[Path, Path]]
