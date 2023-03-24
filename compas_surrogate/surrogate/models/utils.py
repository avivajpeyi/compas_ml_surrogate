import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from corner import corner
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import ShuffleSplit, learning_curve

from compas_surrogate.data_generation.likelihood_cacher import LikelihoodCache
from compas_surrogate.inference_runner import get_training_lnl_cache
from compas_surrogate.plotting.corner import KWGS
from compas_surrogate.surrogate.models import SklearnGPModel


def _get_points_kwgs(color: str, alpha=0.3) -> Dict:
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


def _get_contour_kwgs(color, ls="solid", lw=2, alpha=1.0):
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

    # plot of training and datapoints (no contours)
    fig = corner(training_data[0], **_get_points_kwgs(train_color))
    fig = corner(testing_data[0], **_get_points_kwgs(test_color, 0.85), fig=fig)

    # plot of the training and model contours
    fig = corner(
        training_data[0],
        weights=norm_lnl(training_data[1]),
        fig=fig,
        **_get_contour_kwgs(train_color),
    )
    # plot of the predicted contoursndarray
    fig = corner(
        training_data[0],
        weights=norm_lnl(predictions),
        fig=fig,
        **_get_contour_kwgs(model_color, ls="dashed", lw=1, alpha=1),
    )

    # add legend to figure to right using the following colors
    labels = [
        f"Train ({len(training_data[0])})",
        f"Test ({len(testing_data[0])})",
        "Model",
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


def norm_lnl(lnl: np.ndarray) -> np.array:
    """Normalize the likelihood."""
    lnl = lnl.flatten()
    return np.exp(lnl - np.max(lnl))


def plot_learning_curve(
    title: str,
    in_data: np.ndarray,
    out_data: np.ndarray,
    n_pts: List[int],
    scoring: Optional[str] = "r2",
    axes=None,
) -> plt.Figure:
    """Plot the learning curve for the given model."""

    # collect data
    (train_sizes, train_scores, test_scores, fit_times, pred_times,) = learning_curve(
        estimator=SklearnGPModel().get_model(),
        X=cache.params,
        y=cache.lnl,
        train_sizes=n_pts,
        cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0),
        n_jobs=1,
        scoring=scoring,
        return_times=True,
    )
    train_mu = np.abs(np.mean(train_scores, axis=1))
    train_std = np.abs(np.std(train_scores, axis=1))
    tst_mu = np.abs(np.mean(test_scores, axis=1))
    tst_std = np.abs(np.std(test_scores, axis=1))
    time_mu = np.abs(np.mean(fit_times, axis=1))
    time_std = np.abs(np.std(fit_times, axis=1))
    prd_mu = np.abs(np.mean(pred_times, axis=1))
    prd_std = np.abs(np.std(pred_times, axis=1))

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(5, 10))

    axes[0].set_title(title)
    axes[0].set_xlabel("Training datapoints")
    axes[0].set_ylabel("Score")

    axes[0].grid()
    axes[0].set_yscale("log")
    axes[0].fill_between(
        train_sizes,
        train_mu - train_std,
        train_mu + train_std,
        alpha=0.1,
        color="r",
    )
    axes[0].plot(train_sizes, train_mu, "o-", color="r", label="Training score")
    axes[0].fill_between(
        train_sizes, tst_mu - tst_std, tst_mu + tst_std, alpha=0.1, color="g"
    )
    axes[0].plot(train_sizes, tst_mu, "o-", color="g", label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, time_mu, "o-")
    axes[1].fill_between(train_sizes, time_mu - time_std, time_mu + time_std, alpha=0.1)
    axes[1].set_xlabel("Training datapoints")
    axes[1].set_ylabel("Train time [s]")
    axes[1].set_title("Scalability of the model")

    # Plot n_samples vs pred_time
    axes[2].grid()
    axes[2].plot(train_sizes, prd_mu, "o-")
    axes[2].fill_between(train_sizes, prd_mu - prd_std, prd_mu + prd_std, alpha=0.1)
    axes[2].set_xlabel("Training datapoints")
    axes[2].set_ylabel("Prediction time [s]")
    axes[2].set_title("Performance of the model")

    return axes[0].get_figure()


def plot_learning_curve_for_lnl(outdir, det_matrix_h5, universe_id, n_pts: int = 10):
    os.makedirs(outdir, exist_ok=True)
    cache = get_training_lnl_cache(
        outdir=outdir, det_matrix_h5=det_matrix_h5, universe_id=universe_id
    )
    pts = list(np.linspace(100, 1500, n_pts).astype(int))
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    plot_learning_curve(
        "|R2|", cache.params, cache.lnl, pts, scoring="r2", axes=axes[:, 0]
    )
    plot_learning_curve(
        "MSE",
        cache.params,
        cache.lnl,
        pts,
        scoring="neg_mean_squared_error",
        axes=axes[:, 1],
    )
    fig.tight_layout()
    fig.savefig(f"{outdir}/learning_curve.png")
