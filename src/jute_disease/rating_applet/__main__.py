"""Launch the local sitting instrument."""

import argparse
import csv
import json
from pathlib import Path

from waitress import serve

from jute_disease.rating_applet import create_app, reopen_sitting


def _pool_roots(values: list[str]) -> dict[str, Path]:
    """Parse the configured pool image roots for pair sittings."""
    roots: dict[str, Path] = {}
    for value in values:
        pool, separator, path = value.partition("=")
        if not separator or pool not in {"A", "B", "C"} or not path or pool in roots:
            raise ValueError("--pool-root takes unique POOL=PATH values for A, B, or C")
        roots[pool] = Path(path)
    return roots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open an existing rating sitting locally."
    )
    parser.add_argument("sheet", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--pool-root",
        action="append",
        default=[],
        metavar="POOL=PATH",
        help="Image root for a pair pool; repeat for each pool in the sitting.",
    )
    parser.add_argument(
        "--reopen-reason",
        help="Record an external correction authorization instead of serving.",
    )
    args = parser.parse_args()
    if args.reopen_reason is not None:
        try:
            reopen_sitting(args.sheet, args.reopen_reason)
        except (OSError, ValueError, csv.Error) as error:
            parser.error(f"Unable to reopen sitting: {error}")
        return
    if not args.sheet.is_file():
        parser.error("The sitting sheet does not exist.")
    try:
        pool_roots = _pool_roots(args.pool_root)
    except ValueError as error:
        parser.error(str(error))
    missing_roots = sorted(
        pool for pool, root in pool_roots.items() if not root.is_dir()
    )
    if missing_roots:
        parser.error("Configured pool roots must exist: " + ", ".join(missing_roots))
    with args.sheet.open(newline="", encoding="utf-8") as handle:
        pair_rows = [row for row in csv.DictReader(handle) if row.get("kind") == "pair"]
    if pair_rows:
        try:
            kit = json.loads(
                (args.sheet.parent / "kit.json").read_text(encoding="utf-8")
            )
            item_pools = {
                item["item_id"]: item["pool"]
                for item in kit["items"]
                if item.get("kind") == "pair"
            }
            missing = sorted(
                {
                    item_pools[row["item_id"]]
                    for row in pair_rows
                    if item_pools[row["item_id"]] not in pool_roots
                }
            )
        except OSError, KeyError, TypeError, ValueError:
            parser.error("Pair sitting metadata or configured pool roots are invalid.")
        if missing:
            parser.error(
                "Configure each pair pool with --pool-root POOL=PATH; missing: "
                + ", ".join(missing)
            )
    print(  # noqa: T201
        f"Open http://127.0.0.1:{args.port}/ in your browser. Press Ctrl+C to stop.",
        flush=True,  # noqa: T201
    )
    serve(create_app(args.sheet, pool_roots), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
