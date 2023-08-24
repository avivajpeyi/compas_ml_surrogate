from typing import Dict, Tuple

import numpy as np
from corner import corner
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from compas_surrogate.plotting.corner import KWGS


def plot_model_corner(
    training_data: Tuple[np.ndarray, np.ndarray],
    testing_data: Tuple[np.ndarray, np.ndarray],
    predictions: np.ndarray,
    kwgs={},
) -> plt.Figure:
    """Plot corner plots for the training and testing data."""
    # plot training datapoints

    train_color = kwgs.get("train_col", "tab:blue")
    test_color = kwgs.get("test_col", "tab:orange")
    model_color = kwgs.get("model_col", "tab:green")

    corner_kwgs = dict(
        labels=kwgs.get("labels", None),
        truths=kwgs.get("truths", None),
    )
    # plot of training and datapoints (no contours)
    fig = corner(training_data[0], **__get_points_kwgs(train_color), **corner_kwgs)
    fig = corner(
        testing_data[0], **__get_points_kwgs(test_color, 0.85), fig=fig, **corner_kwgs
    )

    _s = training_data[0][:, 0]
    if len(_s) > 20000:
        bins = 50
    elif len(_s) > 1000:
        bins = 20
    else:
        bins = 10

    # plot of the training and model contours
    fig = corner(
        training_data[0],
        weights=__norm_lnl(training_data[1]),
        fig=fig,
        **__get_contour_kwgs(train_color),
        bins=bins,
    )
    # plot of the predicted contoursndarray
    fig = corner(
        training_data[0],
        weights=__norm_lnl(predictions),
        fig=fig,
        **__get_contour_kwgs(model_color, ls="dashed", lw=1, alpha=1),
        bins=bins,
    )

    # add legend to figure to right using the following colors
    labels = [
        f"Train ({len(training_data[0])})",
        f"Test ({len(testing_data[0])})",
        "Surrogate",
    ]
    colors = [train_color, test_color, model_color]
    legend_elements = [
        Line2D([0], [0], color=c, lw=4, label=l) for c, l in zip(colors, labels)
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.95, 0.95),
        bbox_transform=fig.transFigure,
        frameon=False,
        fontsize=16,
    )

    return fig


def __norm_lnl(lnl: np.ndarray) -> np.array:
    """Normalize the likelihood."""
    lnl = lnl.flatten()
    return np.exp(lnl - np.max(lnl))


def __get_points_kwgs(color: str, alpha=0.3) -> Dict:
    """Get kwargs for corner plot of points."""
    kwgs = KWGS.copy()
    kwgs.update(
        dict(
            plot_datapoints=True,
            plot_contours=False,
            fill_contours=False,
            no_fill_contours=True,
            quantiles=None,
            color=color,
            data_kwargs=dict(alpha=alpha),
            hist_kwargs=dict(alpha=0),
        )
    )
    return kwgs


def __get_contour_kwgs(color, ls="solid", lw=2, alpha=1.0):
    """Get kwargs for corner plot of contours."""
    kwgs = KWGS.copy()
    levels = KWGS.get("levels")
    levels = [levels[1], levels[2]]
    kwgs.update(
        dict(
            plot_datapoints=False,
            plot_contours=True,
            fill_contours=False,
            no_fill_contours=True,
            quantiles=None,
            color=color,
            levels=levels,
            contour_kwargs=dict(linewidths=lw, linestyles=ls, alpha=alpha),
            hist_kwargs=dict(linewidth=lw, linestyle=ls, alpha=alpha),
        )
    )

    return kwgs
