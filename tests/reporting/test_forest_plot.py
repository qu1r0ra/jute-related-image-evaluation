"""The exploratory forest plot: every result outside the prespecified tests."""

from dataclasses import replace
from pathlib import Path

import pytest

from jute_disease.evaluation import (
    PRIMARY_CONTRAST,
    SECONDARY_CONTRASTS,
    Contrast,
    ExploratoryRow,
    Interval,
)
from jute_disease.reporting.forest_plot import (
    EXPLORATORY_TITLE,
    forest_figure,
    write_forest_plot,
)


def _rows(count: int = 4) -> list[ExploratoryRow]:
    return [
        ExploratoryRow(
            label=f"A view 1 route {i}: image-random minus exact-copy",
            pool="A",
            view=1,
            route=f"route {i}",
            minuend="image-random",
            subtrahend="exact-copy",
            method="welch",
            estimate=0.01 * i,
            statistic=1.0,
            df=18.0,
            p_value=0.3,
            interval=Interval(0.01 * i, 0.01 * i - 0.02, 0.01 * i + 0.02, 0.95),
        )
        for i in range(count)
    ]


def test_the_figure_is_labelled_exploratory_and_lists_every_row() -> None:
    rows = _rows()
    axis = forest_figure(rows).axes[0]
    assert "not prespecified" in EXPLORATORY_TITLE
    assert axis.get_title() == EXPLORATORY_TITLE
    labels = [tick.get_text() for tick in axis.get_yticklabels()]
    assert labels == [row.label for row in rows]
    # One interval line and one estimate marker per row, plus the zero line.
    assert len(axis.lines) == 2 * len(rows) + 1


def test_write_forest_plot_saves_an_image(tmp_path: Path) -> None:
    target = write_forest_plot(_rows(), tmp_path / "plots" / "forest.png")
    assert target == tmp_path / "plots" / "forest.png"
    assert target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_only_exploratory_rows_can_be_plotted(tmp_path: Path) -> None:
    prespecified = replace(_rows()[0], status="prespecified")
    with pytest.raises(ValueError, match="exploratory"):
        write_forest_plot([prespecified], tmp_path / "forest.png")
    with pytest.raises(ValueError, match="no rows"):
        write_forest_plot([], tmp_path / "forest.png")
    assert not (tmp_path / "forest.png").exists()


@pytest.mark.parametrize(
    "contrast",
    [
        PRIMARY_CONTRAST,
        *SECONDARY_CONTRASTS.values(),
        Contrast("A", 1, PRIMARY_CONTRAST.route, "image-random", "near-duplicate"),
        Contrast("B", 1, PRIMARY_CONTRAST.route, "image-random", "near-duplicate"),
    ],
)
def test_a_prespecified_contrast_is_refused_whatever_its_status(
    contrast: Contrast, tmp_path: Path
) -> None:
    # The status string is only a label on the row, so it cannot be the guard:
    # a prespecified test mislabelled "exploratory" must still stay off the plot.
    row = ExploratoryRow(
        label=contrast.label,
        pool=contrast.pool,
        view=contrast.view,
        route=contrast.route,
        minuend=contrast.minuend,
        subtrahend=contrast.subtrahend,
        method="welch",
        estimate=0.04,
        statistic=5.0,
        df=18.0,
        p_value=0.001,
        interval=Interval(0.04, 0.02, 0.06, 0.95),
        status="exploratory",
    )
    with pytest.raises(ValueError, match="prespecified"):
        forest_figure([*_rows(), row])
    with pytest.raises(ValueError, match="prespecified"):
        write_forest_plot([row], tmp_path / "forest.png")
    assert not (tmp_path / "forest.png").exists()
