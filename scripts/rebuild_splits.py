"""Rebuild and verify the deterministic partitions in a manuscript rebuild kit."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from jute_disease.reporting import rebuild_kit_splits


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = rebuild_kit_splits(args.kit_manifest, args.image_root, args.output)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"split rebuild blocked: {error}\n")
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
