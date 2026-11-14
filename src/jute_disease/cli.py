"""Single console entry point, ``jute-disease``, for the study workflow.

Each recipe in the justfile is a one-line wrapper over a command below.
Commands marked as planned exit non-zero and name the issue that owns them.
"""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

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
    ROOT_DIR,
    RUNS_DIR,
    SOURCE_INVENTORY_DIR,
)
from jute_disease.data import (
    AUDIT_COUNT,
    DEFAULT_BUDGET_SECONDS,
    DEFAULT_MAX_BYTES,
    OWNER_COUNT,
    SOURCES,
    TARGET_DIR,
    ColorThresholdBackend,
    FreezeBlockedError,
    ImageNetExtractor,
    SceneThresholds,
    SourceMismatchError,
    download_plant_doc,
    download_plant_village,
    evaluate_masks_audit,
    evaluate_ratings,
    export_blind,
    feature_neighbors,
    fit_thresholds_from_manifest,
    fixture_round,
    freeze_target,
    intake_source,
    load_protected_test_images,
    make_synthetic_data,
    prepare_cohort,
    probe_cache,
    read_candidates,
    residual_sample,
    run_masks_audit,
    run_masks_fixture,
    sample_edges,
    synthetic_recall,
    write_policy_manifests,
    write_thresholds,
)
from jute_disease.data.policies import HEALTHY_CAP
from jute_disease.evaluation import (
    FreezeRecordError,
    TestReadRefusedError,
    authorize_test_read,
    default_reads_log_path,
    evaluate_image_diagnostics,
    evaluate_test,
    leaf_sample_ids,
    log_test_read,
    planned_probe_fit_ids,
    verify_freeze_record,
    write_freeze_record,
    write_run_record,
)
from jute_disease.reporting import (
    export_manuscript,
    run_manuscript_fixture,
    validate_claim_audit,
    validate_export,
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
from jute_disease.training.benchmark import load_benchmark
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
        if args.kind == "edges":
            report = sample_edges(
                data, count=args.count, seed=args.seed, threshold=args.threshold
            )
        elif args.kind == "features":
            extractor = ImageNetExtractor()
            report = {
                "pairs": feature_neighbors(data, extractor, k=args.k),
                "feature_provenance": extractor.provenance,
            }
        elif args.kind == "residual":
            extractors: list[ImageNetExtractor] = []

            def extract(paths: Sequence[str]) -> Any:
                extractor = ImageNetExtractor()
                extractors.append(extractor)
                return extractor(paths)

            report = residual_sample(
                data["protected_test"],
                data["training"],
                extract,
                count=args.count,
                k=args.k,
                seed=args.seed,
            )
            report["feature_provenance"] = extractors[-1].provenance
        else:
            images = []
            for path in data:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            report = synthetic_recall(images, threshold=args.threshold)
        _print_json(report)
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"audit blocked: {error}") from error


def _audit_cohort(args: argparse.Namespace) -> None:
    try:
        samples = json.loads(args.source.read_text(encoding="utf-8"))
        cohort = prepare_cohort(samples, seed=args.seed)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "agent.json").write_text(
            json.dumps(cohort, indent=2) + "\n", encoding="utf-8"
        )
        export_blind(cohort["pairs"], args.output / "owner.csv")
        _print_json(
            {
                "pairs": len(cohort["pairs"]),
                "output": str(args.output),
                "next_action": (
                    "Fill agent_rating in agent.json and owner ratings "
                    "in owner.csv independently."
                ),
            }
        )
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
            previous=previous,
            revision=args.revision,
        )
        if previous is not None:
            report["revision"] = args.revision
            report["previous"] = previous
        if args.gate_output:
            gate = {
                key: report[key]
                for key in ("kappa_interval", "threshold", "pass", "method")
            }
            with args.gate_output.open("x", encoding="utf-8") as handle:
                json.dump(gate, handle, indent=2)
                handle.write("\n")
        _print_json(report)
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"audit blocked: {error}") from error


def _masks_calibrate(args: argparse.Namespace) -> None:
    try:
        thresholds = fit_thresholds_from_manifest(
            args.development,
            ColorThresholdBackend(),
            load_protected_test_images(args.target_dir),
        )
        write_thresholds(args.output, thresholds)
    except (ValueError, OSError, KeyError, FreezeBlockedError) as error:
        raise SystemExit(f"mask calibration blocked: {error}") from error
    _print_json(thresholds.to_dict())


def _masks_audit(args: argparse.Namespace) -> None:
    try:
        thresholds = SceneThresholds.from_dict(
            json.loads(args.thresholds.read_text(encoding="utf-8"))
        )
        manifest = run_masks_audit(
            read_candidates(args.inventory),
            thresholds,
            ColorThresholdBackend(),
            args.output,
            load_protected_test_images(args.target_dir),
            seed=args.seed,
            sample_count=args.sample_count,
            owner_count=args.owner_count,
        )
    except (ValueError, OSError, KeyError, FreezeBlockedError) as error:
        raise SystemExit(f"mask audit blocked: {error}") from error
    _print_json(
        {
            "sample_count": manifest["sample_count"],
            "owner_count": manifest["owner_count"],
            "output": str(args.output),
            "sampled_by_scene": manifest["sampling"]["sampled_by_scene"],
            "next_action": (
                "Complete agent.csv. In one blind sitting, rate mask quality and "
                "scene tag in owner.csv; complete the grouping-pair worksheet "
                "afterward."
            ),
        }
    )


def _masks_evaluate(args: argparse.Namespace) -> None:
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        report = evaluate_masks_audit(
            manifest,
            args.agent_sheet,
            args.owner_sheet,
            seed=args.seed,
            agent_rater=(
                {"name": args.agent_name, "version": args.agent_version}
                if args.agent_name and args.agent_version
                else None
            ),
        )
        if args.output:
            with args.output.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
                handle.write("\n")
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"mask evaluation blocked: {error}") from error
    _print_json(report)


def _masks_fixture(args: argparse.Namespace) -> None:
    try:
        _print_json(run_masks_fixture(args.output, seed=args.seed))
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"mask fixture blocked: {error}") from error


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


def _freeze_policies(args: argparse.Namespace) -> None:
    try:
        pools = dict(item.split("=", 1) for item in args.pool)
        if len(pools) != len(args.pool):
            raise ValueError("Pool names must be unique")
        _print_json(
            write_policy_manifests(
                pools,
                args.output,
                args.reserve_fraction,
                args.train_fraction,
                args.val_fraction,
                args.test_fraction,
                args.seed,
                healthy_cap=args.healthy_cap,
            )
        )
    except (ValueError, OSError) as error:
        raise SystemExit(f"policy manifests blocked: {error}") from error


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
            policy_dir=args.policy_dir,
            pool_roots={
                pool: root
                for pool, root in (
                    ("A", args.pool_a_root),
                    ("B", args.pool_b_root),
                    ("C", args.pool_c_root),
                )
                if root is not None
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
    matrix = workspace.benchmark
    if workspace.data.policy_dir is None and (
        len(matrix.pools)
        * len(matrix.policies)
        * len(matrix.views)
        * len(matrix.splits)
        > 1
    ):
        raise SystemExit(
            "The full benchmark needs a receipted policy directory; pass --policy-dir"
        )
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


def _manuscript_fixture(args: argparse.Namespace) -> None:
    try:
        _print_json(run_manuscript_fixture(args.output))
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"manuscript fixture blocked: {error}") from error


def _manuscript_export(args: argparse.Namespace) -> None:
    try:
        result = export_manuscript(
            args.freeze_dir,
            args.config,
            args.output,
            source_metadata=json.loads(
                args.source_metadata.read_text(encoding="utf-8")
            ),
            training_runs_dir=args.training_runs,
            evaluation_metrics_path=args.evaluation_metrics,
            evaluation_artifacts_path=args.evaluation_artifacts,
            statistics_artifacts_path=args.statistics_artifacts,
            figure_sources_path=args.figure_sources,
            export_id=args.export_id,
            command=["jute-disease", *sys.argv[1:]],
            repository_root=args.repository_root,
        )
        _print_json(result)
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"manuscript export blocked: {error}") from error


def _manuscript_check(args: argparse.Namespace) -> None:
    try:
        _print_json(validate_export(args.export_dir))
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"manuscript check blocked: {error}") from error


def _manuscript_audit_check(args: argparse.Namespace) -> None:
    try:
        _print_json(validate_claim_audit(args.export_dir, args.manuscript))
    except (ValueError, OSError, KeyError) as error:
        raise SystemExit(f"manuscript audit blocked: {error}") from error


def _read_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot read {path}: {error}") from error


def _evaluate_probe_plan(args: argparse.Namespace) -> None:
    _print_json(planned_probe_fit_ids(load_benchmark(args.benchmark)))


def _evaluate_freeze_record(args: argparse.Namespace) -> None:
    benchmark = load_benchmark(args.benchmark)
    required = [fit.fit_id for fit in benchmark.fits()] + planned_probe_fit_ids(
        benchmark
    )
    outcomes = _read_json_file(args.outcomes) if args.outcomes else None
    try:
        record = write_freeze_record(
            _read_json_file(args.spec),
            outcomes,
            args.output,
            args.root,
            target_dir=args.target_dir,
            required_fit_ids=required,
            reads_log=args.reads_log,
        )
    except (FreezeRecordError, OSError) as error:
        raise SystemExit(f"freeze record blocked: {error}") from error
    _print_json({"hash": record["hash"], "post_hoc": record["body"]["post_hoc"]})


def _evaluate_run_record(args: argparse.Namespace) -> None:
    try:
        record = write_run_record(
            args.freeze_record,
            _read_json_file(args.runs),
            args.output.parent,
            args.output,
        )
    except (FreezeRecordError, TestReadRefusedError, OSError) as error:
        raise SystemExit(f"run record blocked: {error}") from error
    _print_json({"hash": record["hash"], "fits": sorted(record["body"]["fits"])})


def _evaluate_leaf_sample(args: argparse.Namespace) -> None:
    try:
        record = verify_freeze_record(args.freeze_record, args.root, args.target_dir)
        log_test_read(
            default_reads_log_path(args.root),
            record["hash"],
            [],
            operation="leaf_sample",
        )
        _print_json(list(leaf_sample_ids(args.target_dir, record, args.sample_size)))
    except (TestReadRefusedError, FreezeRecordError, OSError) as error:
        raise SystemExit(f"test read refused: {error}") from error


def _evaluate_test(args: argparse.Namespace) -> None:
    try:
        auth = authorize_test_read(
            args.freeze_record,
            args.run_record,
            args.root,
            args.fit_id or None,
            args.target_dir,
        )
        default_log = default_reads_log_path(args.root)
        fit_ids = [r.fit_id for r in auth.runs]
        additional_logs = [args.reads_log] if args.reads_log else []
        log_test_read(
            default_log, auth.record_hash, fit_ids, additional_logs=additional_logs
        )
        features = _read_json_file(args.features) if args.features else {}
        report = evaluate_test(
            args.target_dir,
            auth,
            args.run_record.parent,
            scenes=features.get("scenes"),
            frame_leaf_share=features.get("frame_leaf_share"),
            sample_size=args.sample_size,
            resamples=args.resamples,
            permutations=args.permutations,
        )
    except (TestReadRefusedError, FreezeRecordError, OSError) as error:
        raise SystemExit(f"test read refused: {error}") from error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + chr(10), encoding="utf-8"
    )
    _print_json({"output": str(args.output), "record_hash": report["record_hash"]})


def _evaluate_diagnostics(args: argparse.Namespace) -> None:
    try:
        auth = authorize_test_read(
            args.freeze_record,
            args.run_record,
            args.root,
            args.fit_id,
            args.target_dir,
        )
        fit_ids = [run.fit_id for run in auth.runs]
        additional_logs = [args.reads_log] if args.reads_log else []
        log_test_read(
            default_reads_log_path(args.root),
            auth.record_hash,
            fit_ids,
            operation="image_diagnostics",
            additional_logs=additional_logs,
        )
        context = _read_json_file(args.context)
        if not isinstance(context, dict):
            raise ValueError("diagnostics context must be a JSON object")
        result = evaluate_image_diagnostics(
            args.target_dir,
            auth,
            args.run_record.parent,
            args.image_root,
            args.output,
            frozen_backbone_checkpoint=args.frozen_backbone_checkpoint,
            assignment_table_path=args.assignment_table,
            model_name=args.model_name,
            scenes=context.get("scenes"),
            policy_predictions=context.get("policy_predictions"),
            sanity_checks=context.get("sanity_checks"),
            generating_command=["jute-disease", *sys.argv[1:]],
            source_artifacts=[
                args.context,
                args.freeze_record,
                args.run_record,
                args.root / "pyproject.toml",
                args.root / "uv.lock",
            ],
        )
    except (TestReadRefusedError, FreezeRecordError, OSError, ValueError) as error:
        raise SystemExit(f"test read refused: {error}") from error
    _print_json({"export": result["export"], "record_hash": auth.record_hash})


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
    parser.add_argument(
        "--policy-dir",
        type=Path,
        default=None,
        help="Receipted pool/view/policy manifests and assignment table.",
    )
    parser.add_argument("--pool-a-root", type=Path, default=None)
    parser.add_argument("--pool-b-root", type=Path, default=None)
    parser.add_argument("--pool-c-root", type=Path, default=None)
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
    sampler.add_argument(
        "--threshold",
        type=int,
        default=4,
        help="Merge threshold for edges and recall (dhash-v2 uses 6).",
    )
    sampler.set_defaults(run=_audit_sample)
    cohort = data_actions.add_parser(
        "audit-cohort",
        help="Select 60 pairs per hash band and write blind rating sheets.",
        description=(
            "Select 60 merged and 60 near-miss pairs, shuffling within "
            "each stratum and cycling through strata using the given seed. "
            "After export, independently complete agent_rating in agent.json "
            "and rating in owner.csv before running audit-evaluate."
        ),
    )
    cohort.add_argument(
        "source", type=Path, help="JSON output from data audit-sample edges"
    )
    cohort.add_argument(
        "output", type=Path, help="New directory for agent.json and owner.csv"
    )
    cohort.add_argument("--seed", type=int, default=DEFAULT_SEED)
    cohort.set_defaults(run=_audit_cohort)
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
    evaluate_audit.add_argument(
        "--gate-output", type=Path, help="Write a new freeze-compatible gate record."
    )
    evaluate_audit.set_defaults(run=_audit_evaluate)

    masks_calibrate = data_actions.add_parser(
        "masks-calibrate",
        help="Fit scene cutoffs on exactly 100 held-apart labeled dev images.",
    )
    masks_calibrate.add_argument("development", type=Path)
    masks_calibrate.add_argument("--output", type=Path, required=True)
    masks_calibrate.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="Frozen target directory whose protected test partition must not "
        "overlap the development set.",
    )
    masks_calibrate.set_defaults(run=_masks_calibrate)

    masks_audit = data_actions.add_parser(
        "masks-audit",
        help="Create stratified mask artifacts and blind agent/owner rating sheets.",
    )
    masks_audit.add_argument("inventory", type=Path)
    masks_audit.add_argument("thresholds", type=Path)
    masks_audit.add_argument("--output", type=Path, required=True)
    masks_audit.add_argument("--seed", type=int, default=DEFAULT_SEED)
    masks_audit.add_argument("--sample-count", type=int, default=AUDIT_COUNT)
    masks_audit.add_argument("--owner-count", type=int, default=OWNER_COUNT)
    masks_audit.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="Frozen target directory whose protected test partition the "
        "development set and audit inventory must not overlap.",
    )
    masks_audit.set_defaults(run=_masks_audit)

    masks_evaluate = data_actions.add_parser(
        "masks-evaluate",
        help="Evaluate blind mask ratings and scene-tag agreement.",
    )
    masks_evaluate.add_argument("manifest", type=Path)
    masks_evaluate.add_argument("agent_sheet", type=Path)
    masks_evaluate.add_argument("owner_sheet", type=Path)
    masks_evaluate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    masks_evaluate.add_argument("--output", type=Path)
    masks_evaluate.add_argument("--agent-name")
    masks_evaluate.add_argument("--agent-version")
    masks_evaluate.set_defaults(run=_masks_evaluate)

    masks_fixture = data_actions.add_parser(
        "masks-fixture",
        help="Run the complete weight-free mask and scene audit fixture.",
    )
    masks_fixture.add_argument("--output", type=Path, required=True)
    masks_fixture.add_argument("--seed", type=int, default=DEFAULT_SEED)
    masks_fixture.set_defaults(run=_masks_fixture)

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
    _add_gated_evaluator(evaluate_actions)

    manuscript = commands.add_parser(
        "manuscript", help="Export and verify manuscript tables, figures and macros."
    )
    manuscript_actions = manuscript.add_subparsers(
        dest="manuscript_action", metavar="ACTION", required=True
    )
    manuscript_fixture = manuscript_actions.add_parser(
        "fixture", help="Run synthetic freeze through export and macro validation."
    )
    manuscript_fixture.add_argument("--output", type=Path, required=True)
    manuscript_fixture.set_defaults(run=_manuscript_fixture)
    manuscript_export = manuscript_actions.add_parser(
        "export", help="Export a verified freeze as manuscript artifacts."
    )
    manuscript_export.add_argument("--freeze-dir", type=Path, required=True)
    manuscript_export.add_argument("--config", type=Path, required=True)
    manuscript_export.add_argument("--output", type=Path, required=True)
    manuscript_export.add_argument("--source-metadata", type=Path, required=True)
    manuscript_export.add_argument("--training-runs", type=Path, required=True)
    manuscript_export.add_argument(
        "--evaluation-metrics",
        type=Path,
        help="Existing validation-metrics aggregate CSV; no evaluation is run.",
    )
    manuscript_export.add_argument(
        "--evaluation-artifacts",
        type=Path,
        help=(
            "Receipt-bound evaluation JSON bundle; no scoring or partition "
            "images are read."
        ),
    )
    manuscript_export.add_argument(
        "--statistics-artifacts",
        type=Path,
        help="Receipt-bound statistics JSON bundle tied to evaluation artifacts.",
    )
    manuscript_export.add_argument("--figure-sources", type=Path)
    manuscript_export.add_argument("--export-id")
    manuscript_export.add_argument(
        "--repository-root",
        type=Path,
        help="Root that every recorded source path is relative to (default: "
        "this repository).",
    )
    manuscript_export.set_defaults(run=_manuscript_export)
    manuscript_check = manuscript_actions.add_parser(
        "check", help="Check exported macros against the authoritative freeze."
    )
    manuscript_check.add_argument("export_dir", type=Path)
    manuscript_check.set_defaults(run=_manuscript_check)
    manuscript_audit = manuscript_actions.add_parser(
        "audit-check", help="Require the completed claim review and owner sign-off."
    )
    manuscript_audit.add_argument("export_dir", type=Path)
    manuscript_audit.add_argument("--manuscript", type=Path)
    manuscript_audit.set_defaults(run=_manuscript_audit_check)

    # planned commands owned by open tickets
    _add_freeze(commands)
    commands.add_parser(
        "profile", help="Profile inference on the desktop (planned)."
    ).set_defaults(run=_planned("profile", "#721"))

    return parser


def _add_gated_evaluator(actions: argparse._SubParsersAction) -> None:
    plan = actions.add_parser("probe-plan", help="List the background and border fits.")
    plan.add_argument("--benchmark", type=Path, default=BENCHMARK_CONFIG)
    plan.set_defaults(run=_evaluate_probe_plan)

    record = actions.add_parser(
        "freeze-record", help="Write the hashed matrix-freeze and selection record."
    )
    record.add_argument("--spec", type=Path, required=True)
    record.add_argument(
        "--outcomes", type=Path, help="Recorded #114 validation outcomes."
    )
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--target-dir", type=Path, required=True)
    record.add_argument("--benchmark", type=Path, default=BENCHMARK_CONFIG)
    record.add_argument("--root", type=Path, default=ROOT_DIR)
    record.add_argument("--reads-log", type=Path)
    record.set_defaults(run=_evaluate_freeze_record)

    runs = actions.add_parser(
        "run-record", help="Bind finished fits to a freeze record."
    )
    runs.add_argument("--freeze-record", type=Path, required=True)
    runs.add_argument("--runs", type=Path, required=True)
    runs.add_argument("--output", type=Path, required=True)
    runs.set_defaults(run=_evaluate_run_record)

    test = actions.add_parser(
        "test", help="Read test predictions; refused without both records."
    )
    test.add_argument("--freeze-record", type=Path)
    test.add_argument("--run-record", type=Path)
    test.add_argument("--target-dir", type=Path, required=True)
    test.add_argument("--output", type=Path, required=True)
    test.add_argument("--features", type=Path, help="Scene tags and frame leaf shares.")
    test.add_argument("--fit-id", action="append")
    test.add_argument("--root", type=Path, default=ROOT_DIR)
    test.add_argument("--reads-log", type=Path)
    test.add_argument("--sample-size", type=int, default=2000)
    test.add_argument("--resamples", type=int, default=10_000)
    test.add_argument("--permutations", type=int, default=1000)
    test.set_defaults(run=_evaluate_test)

    diagnostics = actions.add_parser(
        "diagnostics",
        help="Export gated test-image embeddings and Grad-CAM diagnostics.",
    )
    diagnostics.add_argument("--freeze-record", type=Path, required=True)
    diagnostics.add_argument("--run-record", type=Path, required=True)
    diagnostics.add_argument("--target-dir", type=Path, required=True)
    diagnostics.add_argument("--image-root", type=Path, required=True)
    diagnostics.add_argument("--assignment-table", type=Path, required=True)
    diagnostics.add_argument("--frozen-backbone-checkpoint", type=Path, required=True)
    diagnostics.add_argument(
        "--context",
        type=Path,
        required=True,
        help=(
            "JSON with frozen scene tags, policy predictions and sanity-check records."
        ),
    )
    diagnostics.add_argument("--output", type=Path, required=True)
    diagnostics.add_argument("--fit-id", action="append", required=True)
    diagnostics.add_argument("--model-name", default="mobilenetv2_100.ra_in1k")
    diagnostics.add_argument("--root", type=Path, default=ROOT_DIR)
    diagnostics.add_argument("--reads-log", type=Path)
    diagnostics.set_defaults(run=_evaluate_diagnostics)

    sample = actions.add_parser(
        "leaf-sample", help="Print the stratified leaf-only test sample."
    )
    sample.add_argument("--freeze-record", type=Path)
    sample.add_argument("--target-dir", type=Path, required=True)
    sample.add_argument("--root", type=Path, default=ROOT_DIR)
    sample.add_argument("--sample-size", type=int, default=2000)
    sample.set_defaults(run=_evaluate_leaf_sample)


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
    target.add_argument(
        "--grouping-version", choices=("dhash-v1", "dhash-v2"), default=None
    )
    target.add_argument("--no-merge-ratings", type=Path)
    target.add_argument("--agreement-results", type=Path)
    target.add_argument("--plantvillage-root", type=Path)
    target.add_argument("--plantdoc-root", type=Path)
    target.set_defaults(run=_freeze_target)

    policies = actions.add_parser(
        "policies",
        help="Generate policy-by-view manifests from frozen pools.",
        description=(
            "Each pool is an independently frozen directory. Reserve fractions apply "
            "to per-class families; final fractions apply to non-reserve policy "
            "units. View 2 reserves whole families for its shared test set. "
            "Counts are rounded to the nearest integer; support failures stop output."
        ),
    )
    policies.add_argument(
        "--pool", action="append", required=True, metavar="NAME=FREEZE"
    )
    policies.add_argument("--output", type=Path, required=True)
    policies.add_argument("--reserve-fraction", type=float, required=True)
    policies.add_argument("--train-fraction", type=float, required=True)
    policies.add_argument("--val-fraction", type=float, required=True)
    policies.add_argument("--test-fraction", type=float, required=True)
    policies.add_argument("--seed", type=int, default=DEFAULT_SEED)
    policies.add_argument(
        "--healthy-cap",
        type=int,
        default=HEALTHY_CAP,
        help="Healthy training images kept per policy in pool C (default: %(default)s)",
    )
    policies.set_defaults(run=_freeze_policies)

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
