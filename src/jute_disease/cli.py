"""Single console entry point, ``jute-disease``, for the study workflow.

Each recipe in the justfile is a one-line wrapper over a command below.
Commands marked as planned exit non-zero and name the issue that owns them.
"""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from jute_disease.config import (
    ARTIFACTS_DIR,
    BENCHMARK_CONFIG,
    CURATION_RULES_PATH,
    DEFAULT_SEED,
    FREEZE_DIR,
    IDENTITY_REVIEWS_PATH,
    ML_FEATURES_DIR,
    PLANTDOC_DIR,
    PLANTVILLAGE_DIR,
    RUNS_DIR,
    SOURCE_INVENTORY_DIR,
)
from jute_disease.data import (
    DEFAULT_BUDGET_SECONDS,
    DEFAULT_MAX_BYTES,
    SOURCES,
    TARGET_DIR,
    FreezeBlockedError,
    ImageNetExtractor,
    SourceMismatchError,
    download_plant_doc,
    download_plant_village,
    evaluate_ratings,
    export_blind,
    feature_neighbors,
    fixture_round,
    freeze_target,
    intake_source,
    make_synthetic_data,
    probe_cache,
    residual_sample,
    sample_edges,
    synthetic_recall,
)
from jute_disease.reporting.tables import aggregate_metrics
from jute_disease.training import (
    WORKSPACE_FILE,
    DataInputs,
    Job,
    QueueAbortedError,
    QueueReport,
    RunStore,
    SystemClock,
    Workspace,
    build_plan,
    make_executor,
    make_identity,
    open_workspace,
    run_job,
    run_queue,
    subprocess_launcher,
    verify_stack,
)
from jute_disease.training.engines.ml import predict_ml

Handler = Callable[[argparse.Namespace], None]


def _planned(command: str, owner: str) -> Handler:
    def run(_: argparse.Namespace) -> None:
        raise SystemExit(
            f"jute-disease {command} is not implemented yet; owned by {owner}."
        )

    return run


def _data_download(_: argparse.Namespace) -> None:
    download_plant_village()
    download_plant_doc()


def _data_synthetic(args: argparse.Namespace) -> None:
    plan = make_synthetic_data(args.root, args.per_class, args.size)
    sys.stdout.write(f"Wrote {len(plan.files)} synthetic images under {plan.root}\n")


def _audit_fixture(args: argparse.Namespace) -> None:
    try:
        _print_json(fixture_round(args.output))
    except (ValueError, OSError) as error:
        raise SystemExit(f"audit blocked: {error}") from error


def _audit_sample(args: argparse.Namespace) -> None:
    from PIL import Image

    try:
        data = json.loads(args.source.read_text(encoding="utf-8"))
        extractor = ImageNetExtractor()
        if args.kind == "edges":
            report = sample_edges(data, count=args.count, seed=args.seed)
        elif args.kind == "features":
            report = {
                "pairs": feature_neighbors(data, extractor, k=args.k),
                "feature_provenance": extractor.provenance,
            }
        elif args.kind == "residual":
            report = residual_sample(
                data["protected_test"],
                data["training"],
                extractor,
                count=args.count,
                k=args.k,
                seed=args.seed,
            )
            report["feature_provenance"] = extractor.provenance
        else:
            images = []
            for path in data:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            report = synthetic_recall(images)
        _print_json(report)
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"audit blocked: {error}") from error


def _audit_export(args: argparse.Namespace) -> None:
    try:
        data = json.loads(args.source.read_text(encoding="utf-8"))
        export_blind(data["pairs"], args.sheet)
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"audit blocked: {error}") from error


def _audit_evaluate(args: argparse.Namespace) -> None:
    try:
        data = json.loads(args.source.read_text(encoding="utf-8"))
        previous = (
            json.loads(args.previous.read_text(encoding="utf-8"))
            if args.previous
            else None
        )
        if previous is not None and (
            previous.get("round") != 1
            or previous.get("pass") is not False
            or not args.revision
        ):
            raise ValueError(
                "retry requires a failed first round and a rubric revision"
            )
        if previous is None and args.revision:
            raise ValueError("a rubric revision requires a failed first round")
        report = evaluate_ratings(
            data["pairs"],
            args.sheet,
            data["denominators"],
            round_number=2 if previous else 1,
        )
        if previous is not None:
            report["revision"] = args.revision
            report["previous"] = previous
        _print_json(report)
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"audit blocked: {error}") from error


def _print_json(payload: object) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _freeze_target(args: argparse.Namespace) -> None:
    try:
        report = freeze_target(
            args.output,
            image_root=args.image_root,
            rules_path=args.rules,
            reviews_path=args.reviews,
            seed=args.seed,
            workers=args.workers,
            force=args.force,
            grouping_version=args.grouping_version,
            no_merge_ratings=args.no_merge_ratings,
            agreement_results=args.agreement_results,
            plantvillage_root=args.plantvillage_root,
            plantdoc_root=args.plantdoc_root,
        )
    except (ValueError, OSError) as error:
        raise SystemExit(f"freeze blocked: {error}") from error
    _print_json(report)


def _freeze_cache(args: argparse.Namespace) -> None:
    _print_json(
        probe_cache(
            args.output,
            image_root=args.image_root,
            budget_seconds=args.budget_seconds,
            max_bytes=args.max_bytes,
            workers=args.workers,
        )
    )


def _freeze_source(args: argparse.Namespace) -> None:
    try:
        result = intake_source(
            args.name,
            args.source_root,
            args.output,
            package_version=args.package_version,
            package_digest=args.package_digest,
            leaf_map_path=args.leaf_map,
            seed=args.seed,
            workers=args.workers,
        )
    except (FreezeBlockedError, SourceMismatchError, FileNotFoundError) as error:
        raise SystemExit(f"source intake blocked: {error}") from error
    _print_json(result)


def _workspace_from(args: argparse.Namespace) -> Workspace:
    freeze: Path = args.freeze_dir
    roots = {"plantvillage": args.plantvillage_root, "plantdoc": args.plantdoc_root}
    return open_workspace(
        benchmark_path=args.benchmark,
        data=DataInputs(
            image_root=args.image_root,
            target_dir=freeze / TARGET_DIR,
            sources={
                name: (freeze / f"source_{name}", root) for name, root in roots.items()
            },
        ),
        runs_dir=args.runs_dir,
        cache_dir=args.cache_dir,
        pretrained=args.pretrained,
        require_cuda=not args.allow_cpu,
        accelerator=args.accelerator,
    )


def _train_fit(args: argparse.Namespace) -> None:
    job = Job(
        args.fit_id,
        args.attempt_dir,
        args.result,
        args.upstream_checkpoint,
        args.bounds,
        args.checkpoint_dir,
        args.resume_from,
    )
    run_job(Workspace.load(args.workspace), job)


def _stack(args: argparse.Namespace) -> None:
    _print_json(verify_stack(require_cuda=not args.allow_cpu))


def _queue_plan(args: argparse.Namespace) -> None:
    workspace = _workspace_from(args)
    store = RunStore(workspace.runs_dir)
    _print_json(
        [
            {
                "stage": stage.name,
                "fits": [
                    {"fit": fit.fit_id, "state": store.state(fit.fit_id).value}
                    for fit in stage.fits
                ],
            }
            for stage in build_plan(workspace.benchmark)
        ]
    )


def _queue_status(args: argparse.Namespace) -> None:
    workspace = _workspace_from(args)
    store = RunStore(workspace.runs_dir)
    _print_json(
        {
            fit.fit_id: {
                "state": store.state(fit.fit_id).value,
                "reason": store.reason(fit.fit_id),
            }
            for fit in workspace.benchmark.fits()
        }
    )


def _queue_run(args: argparse.Namespace) -> None:
    workspace = _workspace_from(args)
    workspace_file = workspace.runs_dir / WORKSPACE_FILE
    workspace.save(workspace_file)
    store = RunStore(workspace.runs_dir)
    launcher = subprocess_launcher(workspace_file)
    identity = make_identity(workspace, store)
    try:
        report: QueueReport = run_queue(
            workspace.benchmark,
            store,
            clock=SystemClock(),
            execute=make_executor(workspace, store, identity, launcher),
            identity_for=identity,
            report_path=workspace.runs_dir / "queue.json",
        )
    except QueueAbortedError as error:
        raise SystemExit(f"{error}; see queue.json") from error
    _print_json(
        {
            "outcomes": {k: v.value for k, v in report.outcomes.items()},
            "arms": report.arms,
            "escalations": report.escalations,
        }
    )
    if report.escalations:
        raise SystemExit("queue finished with escalations; see queue.json")


def _evaluate_aggregate(args: argparse.Namespace) -> None:
    aggregate_metrics(args.fit_ids, args.output, args.runs_dir)


def _bounds(text: str) -> tuple[int, int]:
    try:
        train, val = (int(part) for part in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use TRAIN,VAL batch counts") from error
    if train < 1 or val < 1:
        raise argparse.ArgumentTypeError("batch counts must be positive")
    return train, val


def _add_workspace_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK_CONFIG)
    parser.add_argument("--freeze-dir", type=Path, default=FREEZE_DIR)
    parser.add_argument("--image-root", type=Path, default=SOURCE_INVENTORY_DIR)
    parser.add_argument("--plantvillage-root", type=Path, default=PLANTVILLAGE_DIR)
    parser.add_argument("--plantdoc-root", type=Path, default=PLANTDOC_DIR)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=ML_FEATURES_DIR)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Do not require CUDA (fixtures and smoke runs only).",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the declared timm initialization (off for fixtures).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jute-disease",
        description="Jute disease detection: data, training and evaluation.",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    # data: dataset preparation
    data = commands.add_parser("data", help="Prepare the local dataset.")
    data_actions = data.add_subparsers(
        dest="data_action", metavar="ACTION", required=True
    )
    data_actions.add_parser(
        "download", help="Download pinned PlantVillage and PlantDoc packages."
    ).set_defaults(run=_data_download)
    synthetic = data_actions.add_parser(
        "synthetic", help="Write a tiny synthetic dataset for smoke tests."
    )
    synthetic.add_argument("--root", type=Path, default=ARTIFACTS_DIR / "synthetic")
    synthetic.add_argument("--per-class", type=int, default=20)
    synthetic.add_argument("--size", type=int, default=64)
    synthetic.set_defaults(run=_data_synthetic)

    audit = data_actions.add_parser(
        "audit-fixture", help="Run a synthetic blind rating round."
    )
    audit.add_argument("--output", type=Path, required=True)
    audit.set_defaults(run=_audit_fixture)

    sampler = data_actions.add_parser(
        "audit-sample", help="Sample grouping audit candidates from JSON inputs."
    )
    sampler.add_argument("kind", choices=("edges", "features", "residual", "recall"))
    sampler.add_argument("source", type=Path)
    sampler.add_argument("--count", type=int, default=200)
    sampler.add_argument("--seed", type=int, default=DEFAULT_SEED)
    sampler.add_argument("--k", type=int, default=5)
    sampler.set_defaults(run=_audit_sample)
    export = data_actions.add_parser("audit-export", help="Export a blind owner sheet.")
    export.add_argument("source", type=Path)
    export.add_argument("sheet", type=Path)
    export.set_defaults(run=_audit_export)
    evaluate_audit = data_actions.add_parser(
        "audit-evaluate", help="Import owner ratings and evaluate agreement."
    )
    evaluate_audit.add_argument("source", type=Path)
    evaluate_audit.add_argument("sheet", type=Path)
    evaluate_audit.add_argument("--previous", type=Path)
    evaluate_audit.add_argument("--revision")
    evaluate_audit.set_defaults(run=_audit_evaluate)

    # train: the one fit command the queue launches as a child process
    train = commands.add_parser("train", help="Run one benchmark fit.")
    train_actions = train.add_subparsers(
        dest="train_action", metavar="ACTION", required=True
    )
    fit = train_actions.add_parser(
        "fit", help="Run one declared fit; the queue launches this as a child."
    )
    fit.add_argument("--workspace", type=Path, required=True)
    fit.add_argument("--fit-id", required=True)
    fit.add_argument("--attempt-dir", type=Path, required=True)
    fit.add_argument("--result", type=Path, required=True)
    fit.add_argument("--upstream-checkpoint", type=Path, default=None)
    fit.add_argument(
        "--bounds",
        type=_bounds,
        default=None,
        help="TRAIN,VAL batch limits for a timing or smoke run.",
    )
    fit.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Run-level directory that keeps the full training state.",
    )
    fit.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="A last.ckpt in --checkpoint-dir to continue training from.",
    )
    fit.set_defaults(run=_train_fit)

    stack = commands.add_parser("stack", help="Verify the measured CUDA stack.")
    stack.add_argument("--allow-cpu", action="store_true")
    stack.set_defaults(run=_stack)

    # queue: the serial, resumable run of the whole benchmark
    queue = commands.add_parser("queue", help="Plan, run and inspect the queue.")
    queue_actions = queue.add_subparsers(
        dest="queue_action", metavar="ACTION", required=True
    )
    plan = queue_actions.add_parser("plan", help="Show stages and current run states.")
    _add_workspace_options(plan)
    plan.set_defaults(run=_queue_plan)
    status = queue_actions.add_parser("status", help="Show every fit's state.")
    _add_workspace_options(status)
    status.set_defaults(run=_queue_status)
    run = queue_actions.add_parser("run", help="Run or resume the benchmark.")
    _add_workspace_options(run)
    run.set_defaults(run=_queue_run)

    # evaluate: aggregation and inference
    evaluate = commands.add_parser("evaluate", help="Aggregate and predict.")
    evaluate.set_defaults(run=_planned("evaluate", "#721"))
    evaluate_actions = evaluate.add_subparsers(dest="evaluate_action", metavar="ACTION")
    aggregate = evaluate_actions.add_parser(
        "aggregate", help="Write the experiment summary CSV."
    )
    aggregate.add_argument(
        "--exp-names",
        dest="fit_ids",
        nargs="+",
        required=True,
        help="Fit ids, for example cnn/mobilenet_v2_direct/seed-42.",
    )
    aggregate.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    aggregate.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_DIR / "experiment_summary.csv",
        help="Output CSV path.",
    )
    aggregate.set_defaults(run=_evaluate_aggregate)
    evaluate_actions.add_parser("predict", help="Classify one image (forwarded).")

    # planned commands owned by open tickets
    _add_freeze(commands)
    commands.add_parser(
        "profile", help="Profile inference on the desktop (planned)."
    ).set_defaults(run=_planned("profile", "#721"))

    return parser


def _add_freeze(commands: argparse._SubParsersAction) -> None:
    freeze = commands.add_parser(
        "freeze", help="Freeze the target dataset, its cache decision and sources."
    )
    actions = freeze.add_subparsers(
        dest="freeze_action", metavar="ACTION", required=True
    )

    def common(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--output", type=Path, default=FREEZE_DIR)
        parser.add_argument("--workers", type=int, default=4)

    target = actions.add_parser(
        "target", help="Fingerprint, curate, resolve identity and write the manifest."
    )
    common(target)
    target.add_argument("--image-root", type=Path, default=SOURCE_INVENTORY_DIR)
    target.add_argument("--rules", type=Path, default=CURATION_RULES_PATH)
    target.add_argument("--reviews", type=Path, default=IDENTITY_REVIEWS_PATH)
    target.add_argument("--seed", type=int, default=DEFAULT_SEED)
    target.add_argument(
        "--force", action="store_true", help="Replace a different completed freeze."
    )
    target.add_argument("--grouping-version", choices=("dhash-v1",), default=None)
    target.add_argument("--no-merge-ratings", type=Path)
    target.add_argument("--agreement-results", type=Path)
    target.add_argument("--plantvillage-root", type=Path)
    target.add_argument("--plantdoc-root", type=Path)
    target.set_defaults(run=_freeze_target)

    cache = actions.add_parser(
        "cache", help="Bounded train/val cache probe and its cache decision."
    )
    common(cache)
    cache.add_argument("--image-root", type=Path, default=SOURCE_INVENTORY_DIR)
    cache.add_argument("--budget-seconds", type=float, default=DEFAULT_BUDGET_SECONDS)
    cache.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    cache.set_defaults(run=_freeze_cache)

    source = actions.add_parser(
        "source", help="Record a pinned source package and check it against the target."
    )
    common(source)
    source.add_argument("name", choices=sorted(SOURCES))
    source.add_argument("source_root", type=Path)
    source.add_argument("--package-version", type=int, required=True)
    source.add_argument(
        "--package-digest",
        required=True,
        help="Expected inventory SHA-256 for the pinned source version",
    )
    source.add_argument("--leaf-map", type=Path, default=None)
    source.add_argument("--seed", type=int, default=DEFAULT_SEED)
    source.set_defaults(run=_freeze_source)


def main(argv: Sequence[str] | None = None) -> None:
    tokens = list(sys.argv[1:] if argv is None else argv)
    # predict owns its argument parsing, so its arguments bypass argparse.
    if tuple(tokens[:2]) == ("evaluate", "predict"):
        predict_ml(tokens[2:])
    else:
        args = build_parser().parse_args(tokens)
        args.run(args)


if __name__ == "__main__":
    main()
