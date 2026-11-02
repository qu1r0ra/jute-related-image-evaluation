"""Figures, tables and reproducible manuscript exports for the study."""

from jute_disease.reporting.manuscript import (
    export_manuscript,
    rebuild_kit_splits,
    run_manuscript_fixture,
    validate_claim_audit,
    validate_export,
    validate_figure_sources,
)
from jute_disease.reporting.tables import (
    aggregate_metric_rows,
    aggregate_metrics,
    latest_metrics_path,
)

__all__ = [
    "export_manuscript",
    "aggregate_metric_rows",
    "aggregate_metrics",
    "latest_metrics_path",
    "rebuild_kit_splits",
    "run_manuscript_fixture",
    "validate_claim_audit",
    "validate_export",
    "validate_figure_sources",
]
