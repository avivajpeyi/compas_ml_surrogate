import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import ShuffleSplit, learning_curve

from compas_surrogate.inference_runner import (
    get_ml_surrogate_model,
    get_training_lnl_cache,
)
from compas_surrogate.surrogate.models import SklearnGPModel
OUTDIR = "out_learning_curve"
H5 = "det_matrix.h5"
random.seed(1)


def plot_learning_curve(title, cache, n_pts, scoring, axes=None):
    # collect data
    (
        train_sizes,
        train_scores,
        test_scores,
        fit_times,
        pred_times,
    ) = learning_curve(
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

    # plot
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

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
    axes[0].plot(
        train_sizes, train_mu, "o-", color="r", label="Training score"
    )
    axes[0].fill_between(
        train_sizes, tst_mu - tst_std, tst_mu + tst_std, alpha=0.1, color="g"
    )
    axes[0].plot(
        train_sizes, tst_mu, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, time_mu, "o-")
    axes[1].fill_between(
        train_sizes, time_mu - time_std, time_mu + time_std, alpha=0.1
    )
    axes[1].set_xlabel("Training datapoints")
    axes[1].set_ylabel("Train time [s]")
    axes[1].set_title("Scalability of the model")

    # Plot n_samples vs pred_time
    axes[2].grid()
    axes[2].plot(train_sizes, prd_mu, "o-")
    axes[2].fill_between(
        train_sizes, prd_mu - prd_std, prd_mu + prd_std, alpha=0.1
    )
    axes[2].set_xlabel("Training datapoints")
    axes[2].set_ylabel("Prediction time [s]")
    axes[2].set_title("Performance of the model")

    return axes[0].get_figure()


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    cache = get_training_lnl_cache(
        outdir=OUTDIR, det_matrix_h5=H5, universe_id=5000
    )
    n_pts = np.linspace(100, 1500, 15).astype(int)
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    plot_learning_curve("|R2|", cache, n_pts, scoring="r2", axes=axes[:, 0])
    plot_learning_curve(
        "MSE", cache, n_pts, scoring="neg_mean_squared_error", axes=axes[:, 1]
    )
    fig.tight_layout()
    fig.savefig(f"{OUTDIR}/learning_curve.png")


if __name__ == "__main__":
    main()
