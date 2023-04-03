import matplotlib.pyplot as plt
import numpy as np

from compas_surrogate.data_generation.likelihood_cacher import (
    Universe,
    get_training_lnl_cache,
)
from compas_surrogate.plotting.hist2d import plot_probs
from compas_surrogate.utils import exp_norm_scale_log_data

OUT = "lnl_plots"

mock_uni = Universe.from_hdf5("det_matrix.h5", 10)


def make_plots(det_fname, mock_uni, outdir):
    cache = get_training_lnl_cache(
        outdir=outdir, det_matrix_h5=det_fname, mock_uni=mock_uni, clean=False
    )

    cache_subset = cache.dataframe
    norm_p = exp_norm_scale_log_data(cache_subset["lnl"])

    LEVELS = np.quantile(norm_p, [0.1, 0.5, 0.68, 0.95, 1])
    true_params = [cache.true_dict["muz"], cache.true_dict["sigma0"]]
    fig, ax = plot_probs(
        cache_subset["muz"],
        cache_subset["sigma0"],
        norm_p,
        levels=LEVELS,
        cmap="Oranges",
        true_values=true_params,
    )
    fig.savefig(f"{outdir}/lnl.png", dpi=300)

    fig, ax = plot_probs(
        cache_subset["muz"],
        cache_subset["sigma0"],
        norm_p,
        levels=LEVELS,
        cmap="Oranges",
        true_values=true_params,
        zoom_range=[-0.5, -0.45, 0.12, 0.26],
    )
    fig.savefig(f"{outdir}/lnl_zoom.png", dpi=300)

    fig, ax = plt.subplots(1, 1)
    x_range = [-0.471, -0.469]
    cache_subset = cache_subset[
        (cache_subset["muz"] > x_range[0]) & (cache_subset["muz"] < x_range[1])
    ]
    norm_p = exp_norm_scale_log_data(cache_subset["lnl"])
    y_range = [0.12, 0.26]
    y_bins = np.linspace(y_range[0], y_range[1], 50)
    lnl_in_sigma_bins = []
    for i in range(len(y_bins) - 1):
        y = norm_p[
            (cache_subset["sigma0"] > y_bins[i])
            & (cache_subset["sigma0"] < y_bins[i + 1])
        ]
        lnl_in_sigma_bins.append(np.sum(y))
    lnl_in_sigma_bins = np.array(lnl_in_sigma_bins)
    lnl_in_sigma_bins = lnl_in_sigma_bins / np.sum(lnl_in_sigma_bins)
    ax.plot(y_bins[:-1], lnl_in_sigma_bins)
    ax.set_xlabel("sigma0")
    ax.set_ylabel("lnl")
    fig.savefig(f"{outdir}/lnl_zoom_sigma.png", dpi=300)


make_plots("det_matrix.h5", mock_uni, "lnl_plots")
make_plots("focused_data.h5", mock_uni, "lnl_plots_focused")
