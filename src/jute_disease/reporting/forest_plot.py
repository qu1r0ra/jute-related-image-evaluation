"""Forest plot of the exploratory contrasts (ADR 0004).

Every result outside the prespecified tests lands here, labelled exploratory:
the plot shows each estimate and interval but supports no confirmatory claim.
"""

from collections.abc import Sequence
from pathlib import Path

from matplotlib.figure import Figure

from jute_disease.evaluation import ExploratoryRow, is_prespecified

EXPLORATORY_TITLE = "Exploratory contrasts, not prespecified (macro-F1 difference)"


def forest_figure(rows: Sequence[ExploratoryRow]) -> Figure:
    """Draw one estimate-and-interval line per exploratory row, top to bottom.

    A row is refused if its contrast is one of the prespecified tests (the
    primary, the secondaries and the A/B equivalence contrasts), whatever its
    ``status`` label says: the label is data on the row, the contrast is what it
    compares.
    """
    if not rows:
        raise ValueError("no rows to plot")
    if any(row.status != "exploratory" for row in rows):
        raise ValueError("the forest plot holds exploratory rows only")
    held = [row.label for row in rows if is_prespecified(row.contrast)]
    if held:
        raise ValueError(f"prespecified contrasts stay off the forest plot: {held}")
    figure = Figure(figsize=(8, 0.3 * len(rows) + 1.5), layout="constrained")
    axis = figure.subplots()
    positions = list(range(len(rows)))[::-1]
    for position, row in zip(positions, rows, strict=True):
        axis.plot(
            [row.interval.lower, row.interval.upper], [position, position], color="0.3"
        )
        axis.plot(row.estimate, position, "o", color="C0")
    axis.axvline(0, color="0.6", linestyle="--", linewidth=1)
    axis.set_yticks(positions)
    axis.set_yticklabels([row.label for row in rows], fontsize=7)
    axis.set_xlabel(f"Difference in macro-F1 ({rows[0].interval.level:.0%} interval)")
    axis.set_title(EXPLORATORY_TITLE)
    return figure


def write_forest_plot(rows: Sequence[ExploratoryRow], path: Path) -> Path:
    """Save the exploratory forest plot as a PNG and return its path."""
    figure = forest_figure(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, format="png")
    return path
