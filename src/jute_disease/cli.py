"""Single console entry point, ``jute-disease``, for the study workflow.

Each recipe in the justfile is a one-line wrapper over a command below.
Commands marked as planned exit non-zero and name the issue that owns them.
"""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from dotenv import load_dotenv
from lightning.pytorch.cli import LightningCLI

from jute_disease.config import (
    ARTIFACTS_DIR,
    CURATION_RULES_PATH,
    DEFAULT_SEED,
    FREEZE_DIR,
    IDENTITY_REVIEWS_PATH,
    ML_MODELS_DIR,
    SOURCE_INVENTORY_DIR,
)
from jute_disease.data import (
    DEFAULT_BUDGET_SECONDS,
    DEFAULT_MAX_BYTES,
    SOURCES,
    DataModule,
    FreezeBlockedError,
    SourceMismatchError,
    download_plant_doc,
    download_plant_village,
    freeze_target,
    intake_source,
    make_synthetic_data,
    probe_cache,
)
from jute_disease.models.dl.classifier import Classifier
from jute_disease.models.ml import FEATURE_EXTRACTORS, ML_CLASSIFIERS
from jute_disease.reporting.tables import aggregate_metrics
from jute_disease.training import check_deadline, pretrain_main, with_training_config
from jute_disease.training.baselines import CONFIGS_DIR, run_all_dl
from jute_disease.training.cross_validation import run_cross_validation
from jute_disease.training.engines.ml.predict import predict_ml
from jute_disease.training.engines.ml.sweep import run_all_ml
from jute_disease.training.engines.ml.train import train_ml
from jute_disease.training.grid_search import run_grid_search
from jute_disease.training.partitions import HELD_OUT_SUBCOMMANDS
from jute_disease.training.queue import (
    QUEUE_DIR,
    RECIPES_DIR,
    QueueEscalationError,
    QueueNotAdmittedError,
    build_plan,
    plan_from_json,
    plan_to_json,
    run_queue,
)

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


def _print_json(payload: object) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _queue_check(args: argparse.Namespace) -> None:
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        _print_json(check_deadline(config, args.now))
    except (OSError, ValueError) as error:
        raise SystemExit(f"queue blocked: {error}") from error


def _queue_plan(args: argparse.Namespace) -> None:
    """Write the stage plan; planning never starts or admits a run."""
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        plan = build_plan(
            config,
            args.seed,
            recipes_dir=args.recipes_dir,
            state_dir=args.state_dir,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"queue blocked: {error}") from error
    payload = plan_to_json(plan, args.seed)
    if args.output is None:
        _print_json(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        sys.stdout.write(f"Wrote {len(plan)} planned stages to {args.output}\n")


def _queue_run(args: argparse.Namespace) -> None:
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        plan = plan_from_json(json.loads(args.plan.read_text(encoding="utf-8")))
        _print_json(run_queue(plan, args.state_dir, config, admit=args.admit))
    except QueueNotAdmittedError as error:
        raise SystemExit(f"queue not admitted: {error}; pass --admit") from error
    except QueueEscalationError as error:
        raise SystemExit(f"queue escalation: {error}") from error
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"queue blocked: {error}") from error


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
        )
    except ValueError as error:
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


def _train_dl(lightning_args: list[str]) -> None:
    if held_out := HELD_OUT_SUBCOMMANDS.intersection(lightning_args):
        raise SystemExit(
            f"train dl {sorted(held_out)[0]} would read the held-out test "
            "partition; scoring it is owned by the final evaluator (#721)."
        )
    load_dotenv()
    LightningCLI(
        model_class=Classifier,
        datamodule_class=DataModule,
        save_config_kwargs={"overwrite": True},
        args=with_training_config(lightning_args),
    )


def _train_ml(args: argparse.Namespace) -> None:
    if args.all:
        run_all_ml(args.classifiers, args.feature_types, balanced=args.balanced)
    else:
        train_ml(
            classifier=args.classifier,
            feature_type=args.feature_type,
            balanced=args.balanced,
            seed=args.seed,
            output_dir=args.output_dir,
        )


def _train_baselines(args: argparse.Namespace) -> None:
    run_all_dl(configs_dir=args.configs_dir, config_file=args.config)


def _train_cv(args: argparse.Namespace) -> None:
    run_cross_validation(args.config, args.folds)


def _train_grid(args: argparse.Namespace) -> None:
    run_grid_search(args.grid_config, args.base_config, args.fast_dev_run)


def _evaluate_aggregate(args: argparse.Namespace) -> None:
    aggregate_metrics(args.exp_names, args.output)


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

    # train: model training and experiment matrices
    train = commands.add_parser("train", help="Train models and run experiments.")
    train_actions = train.add_subparsers(
        dest="train_action", metavar="ACTION", required=True
    )
    # Forwarded commands: main() passes their arguments through unparsed.
    train_actions.add_parser(
        "dl", help="Lightning CLI for DL (e.g. dl fit --config ...)."
    )

    ml = train_actions.add_parser(
        "ml", help="Train one classical model, or every combination with --all."
    )
    ml.add_argument("--all", action="store_true", help="Run every combination.")
    ml.add_argument(
        "--classifier",
        default="rf",
        choices=list(ML_CLASSIFIERS),
        help="Classifier to train (single run).",
    )
    ml.add_argument(
        "--feature_type",
        default="crafted",
        choices=list(FEATURE_EXTRACTORS),
        help="Feature type to extract (single run).",
    )
    ml.add_argument(
        "--classifiers",
        nargs="+",
        default=list(ML_CLASSIFIERS),
        choices=list(ML_CLASSIFIERS),
        help="Classifiers for --all.",
    )
    ml.add_argument(
        "--feature-types",
        nargs="+",
        default=list(FEATURE_EXTRACTORS),
        choices=list(FEATURE_EXTRACTORS),
        help="Feature types for --all.",
    )
    ml.add_argument(
        "--balanced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use balanced sample weights.",
    )
    ml.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ml.add_argument(
        "--output-dir", type=Path, default=ML_MODELS_DIR, help="Model directory."
    )
    ml.set_defaults(run=_train_ml)

    baselines = train_actions.add_parser(
        "baselines", help="Fit every DL baseline config (no test scoring)."
    )
    baselines.add_argument("--configs-dir", type=Path, default=CONFIGS_DIR)
    baselines.add_argument(
        "--config", type=Path, default=None, help="Run a single baseline config."
    )
    baselines.set_defaults(run=_train_baselines)

    cv = train_actions.add_parser("cv", help="K-fold cross-validation for one config.")
    cv.add_argument("config", type=Path)
    cv.add_argument(
        "--folds", type=int, default=None, help="Overrides k_fold in the config."
    )
    cv.set_defaults(run=_train_cv)

    grid = train_actions.add_parser("grid", help="Run a DL grid search.")
    grid.add_argument("grid_config", help="Grid search YAML.")
    grid.add_argument(
        "--base-config", default=None, help="Base model YAML (default: match name)."
    )
    grid.add_argument(
        "--fast-dev-run", action="store_true", help="One batch per stage."
    )
    grid.set_defaults(run=_train_grid)

    train_actions.add_parser("pretrain", help="Domain pretraining (forwarded).")

    # evaluate: aggregation, classical test evaluation and inference
    evaluate = commands.add_parser("evaluate", help="Aggregate, evaluate and predict.")
    evaluate.set_defaults(run=_planned("evaluate", "#721"))
    evaluate_actions = evaluate.add_subparsers(dest="evaluate_action", metavar="ACTION")
    aggregate = evaluate_actions.add_parser(
        "aggregate", help="Write the experiment summary CSV."
    )
    aggregate.add_argument(
        "--exp-names", nargs="+", required=True, help="Experiment log names."
    )
    aggregate.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_DIR / "experiment_summary.csv",
        help="Output CSV path.",
    )
    aggregate.set_defaults(run=_evaluate_aggregate)

    evaluate_ml = evaluate_actions.add_parser(
        "ml", help="Test one trained classical model (final evaluator, planned)."
    )
    evaluate_ml.add_argument("--classifier", default="rf", choices=list(ML_CLASSIFIERS))
    evaluate_ml.add_argument(
        "--feature_type", default="crafted", choices=list(FEATURE_EXTRACTORS)
    )
    evaluate_ml.add_argument("--seed", type=int, default=DEFAULT_SEED)
    evaluate_ml.set_defaults(run=_planned("evaluate ml", "#721"))

    evaluate_actions.add_parser("predict", help="Classify one image (forwarded).")

    # planned commands owned by open tickets
    _add_freeze(commands)
    queue = commands.add_parser(
        "queue", help="Plan, check and run the resumable training queue."
    )
    queue_actions = queue.add_subparsers(
        dest="queue_action", metavar="ACTION", required=True
    )
    plan = queue_actions.add_parser(
        "plan", help="Write the ordered stage plan; starts nothing."
    )
    plan.add_argument("--config", type=Path, required=True)
    plan.add_argument("--recipes-dir", type=Path, default=RECIPES_DIR)
    plan.add_argument("--state-dir", type=Path, default=QUEUE_DIR)
    plan.add_argument("--seed", type=int, default=DEFAULT_SEED)
    plan.add_argument("--output", type=Path, default=None)
    plan.set_defaults(run=_queue_plan)
    run = queue_actions.add_parser(
        "run", help="Run or resume a plan; requires an explicit --admit."
    )
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--state-dir", type=Path, default=QUEUE_DIR)
    run.add_argument(
        "--admit", action="store_true", help="Admit the run; never implied."
    )
    run.set_defaults(run=_queue_run)
    check = queue_actions.add_parser(
        "check", help="Check a simulated deadline forecast."
    )
    check.add_argument("--config", type=Path, required=True)
    check.add_argument("--now", required=True)
    check.set_defaults(run=_queue_check)
    commands.add_parser(
        "profile", help="Profile inference on the desktop (planned)."
    ).set_defaults(run=_planned("profile", "#721"))

    return parser


def _add_freeze(commands: "argparse._SubParsersAction") -> None:
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
    head = tuple(tokens[:2])
    # These commands own their argument parsing, so their arguments bypass argparse.
    if head == ("train", "dl"):
        _train_dl(tokens[2:])
    elif head == ("train", "pretrain"):
        pretrain_main(tokens[2:])
    elif head == ("evaluate", "predict"):
        predict_ml(tokens[2:])
    else:
        args = build_parser().parse_args(tokens)
        args.run(args)


if __name__ == "__main__":
    main()
