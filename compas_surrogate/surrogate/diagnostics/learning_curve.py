import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import ShuffleSplit, learning_curve
from sklearn.preprocessing import StandardScaler

from compas_surrogate.data_generation.likelihood_cacher import get_training_lnl_cache
from compas_surrogate.surrogate.models import SklearnGPModel


def plot_learning_curve_for_lnl(
    outdir, det_matrix_h5, universe_id, training_sizes=None, verbose=2
):
    os.makedirs(outdir, exist_ok=True)
    cache = get_training_lnl_cache(
        outdir=outdir, det_matrix_h5=det_matrix_h5, universe_id=universe_id
    )
    if training_sizes is None:
        training_sizes = list(np.linspace(100, 1000, 10).astype(int))
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    __plot_learning_curve(
        "|R2|",
        cache.params,
        cache.lnl,
        training_sizes,
        scoring="r2",
        axes=axes[:, 0],
        verbose=verbose,
    )
    __plot_learning_curve(
        "MSE",
        cache.params,
        cache.lnl,
        training_sizes,
        scoring="neg_mean_squared_error",
        axes=axes[:, 1],
        verbose=verbose,
    )
    fig.tight_layout()
    fig.savefig(f"{outdir}/learning_curve.png")


def __plot_learning_curve(
    title: str,
    in_data: np.ndarray,
    out_data: np.ndarray,
    n_pts: List[int],
    scoring: Optional[str] = "r2",
    axes=None,
    verbose=2,
) -> plt.Figure:
    """Plot the learning curve for the given model."""

    scaler = StandardScaler()
    in_scaled = scaler.fit_transform(in_data)

    # collect data
    (train_sizes, train_scores, test_scores, fit_times, pred_times,) = learning_curve(
        estimator=SklearnGPModel().get_model(),
        X=in_scaled,
        y=out_data,
        train_sizes=n_pts,
        cv=ShuffleSplit(n_splits=3, test_size=0.2, random_state=0),
        n_jobs=2,
        scoring=scoring,
        return_times=True,
        verbose=verbose,
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
