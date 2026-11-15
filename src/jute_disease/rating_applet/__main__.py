"""Launch the local sitting instrument."""

import argparse
from pathlib import Path

from waitress import serve

from jute_disease.rating_applet import create_app, reopen_sitting


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open an existing rating sitting locally."
    )
    parser.add_argument("sheet", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--reopen-reason",
        help="Record an external correction authorization instead of serving.",
    )
    args = parser.parse_args()
    if args.reopen_reason is not None:
        reopen_sitting(args.sheet, args.reopen_reason)
        return
    if not args.sheet.is_file():
        parser.error("The sitting sheet does not exist.")
    print(  # noqa: T201
        f"Open http://127.0.0.1:{args.port}/ in your browser. Press Ctrl+C to stop.",
        flush=True,  # noqa: T201
    )
    serve(create_app(args.sheet), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
