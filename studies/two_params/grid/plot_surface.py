import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata

from compas_surrogate.data_generation.likelihood_cacher import (
    Universe,
    get_training_lnl_cache,
)
from compas_surrogate.plotting.image_utils import horizontal_concat, vertical_concat


def plot_1d(cache, true_val):
    cache_subset = cache.dataframe
    z = cache_subset["lnl"]
    x, y = cache_subset["muz"].values, cache_subset["sigma0"].values
    data = pd.DataFrame({"muz": x, "sigma0": y, "z": z})
    data = data[data["muz"] < -0.35]

    num_x, num_y = len(np.unique(x)), len(np.unique(y))
    xi = np.linspace(data.muz.min(), data.muz.max(), num_x)
    yi = np.linspace(data.sigma0.min(), data.sigma0.max(), num_y)
    zi_at_true_y = griddata(
        (data.muz, data.sigma0),
        data.z,
        (xi[None, :], true_val["sigma0"]),
        method="linear",
        fill_value=-np.inf,
    )[0]
    zi_at_true_y = zi_at_true_y - zi_at_true_y.max()

    zi_at_true_x = griddata(
        (data.muz, data.sigma0),
        data.z,
        (true_val["muz"], yi[None, :]),
        method="linear",
        fill_value=-np.inf,
    )[0]
    zi_at_true_x = zi_at_true_x - zi_at_true_x.max()

    # plot
    fig, axes = plt.subplots(2, 1, figsize=(4, 5))
    axes[0].plot(xi, zi_at_true_y)
    axes[0].axvline(true_val["muz"], color="k", linestyle="--", lw=0.5)
    axes[0].set_ylabel("lnL-max(lnL)")
    axes[0].set_xlabel("muz")
    axes[1].plot(yi, zi_at_true_x)
    axes[1].axvline(true_val["sigma0"], color="k", linestyle="--", lw=0.5)
    axes[1].set_ylabel("lnL-max(lnL)")
    axes[1].set_xlabel("sigma0")
    return fig


def make_plots(cache, mock_uni, label=""):
    cache_subset = cache.dataframe
    z = cache_subset["lnl"] - cache_subset["lnl"].max()
    x, y = cache_subset["muz"].values, cache_subset["sigma0"].values
    data = pd.DataFrame({"x": x, "y": y, "z": z})
    num_x, num_y = len(np.unique(x)), len(np.unique(y))
    plt_range = [data.x.min(), data.x.max(), y.min(), y.max()]
    true_x, true_y = mock_uni.muz, mock_uni.sigma0

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    cmap_args = dict(cmap="RdBu", vmin=-3, vmax=0)
    xi = np.linspace(data.x.min(), data.x.max(), num_x)
    yi = np.linspace(data.y.min(), data.y.max(), num_y)
    zi = griddata((data.x, data.y), data.z, (xi[None, :], yi[:, None]), method="linear")
    cb = ax.pcolormesh(xi, yi, zi, **cmap_args)
    fig.colorbar(cb, ax=ax, label="lnL-max(lnL)")
    ax.axis(plt_range)
    ax.set_xlabel("muz")
    ax.set_ylabel("sigma0")
    ax.scatter(true_x, true_y, marker="+", color="k", s=100)
    ax.axvline(true_x, color="k", linestyle="--", lw=0.5)
    ax.axhline(true_y, color="k", linestyle="--", lw=0.5)
    fig.suptitle(f"{label}")
    plt.tight_layout()
    return fig, plt_range


def main_plotter(large_grid, zoom_in_grid, idx):
    mock_uni = Universe.from_hdf5(zoom_in_grid, idx)
    mock_population = mock_uni.sample_possible_event_matrix()

    outdir = f"mock_uni_{idx}"
    large_cache = get_training_lnl_cache(
        outdir=outdir + "/large_cache",
        det_matrix_h5=large_grid,
        mock_uni=mock_population.universe,
        clean=False,
    )
    zoom_cache = get_training_lnl_cache(
        outdir=outdir + "/zoom_cache",
        det_matrix_h5=zoom_in_grid,
        mock_uni=mock_population.universe,
        clean=False,
    )
    fig_full, _ = make_plots(large_cache, mock_population.universe, label="Full range")
    fig_zoom, zoom_range = make_plots(
        zoom_cache, mock_population.universe, label="Zoom"
    )
    ax = fig_full.get_axes()[0]
    ax.add_patch(
        Rectangle(
            xy=(zoom_range[0], zoom_range[2]),  # point of origin.
            width=zoom_range[1] - zoom_range[0],
            height=zoom_range[3] - zoom_range[2],
            linewidth=1,
            color="black",
            fill=False,
        )
    )
    fig_full.savefig(f"{outdir}/lnl_full.png")
    ax = fig_full.get_axes()[0]
    ax.set_xlim(-0.5, -0.4)
    ax.set_ylim(0.1, 0.3)
    fig_full.suptitle("Zoomed in")
    fig_full.savefig(f"{outdir}/lnl_zoom.png")

    fig_zoom.suptitle("Zoomed in (high-res)")
    fig_zoom.savefig(f"{outdir}/lnl_highres.png")

    horizontal_concat(
        [
            f"{outdir}/lnl_full.png",
            f"{outdir}/lnl_zoom.png",
            f"{outdir}/lnl_highres.png",
        ],
        f"{outdir}/lnl.png",
        rm_orig=True,
    )

    max_lnl_param = large_cache.max_lnl_param_dict()

    max_lnl_uni = Universe.from_hdf5(large_grid, search_param=max_lnl_param)
    fig: plt.Figure = max_lnl_uni.plot_detection_rate_matrix(
        save=False, scatter_events=mock_population.mcz
    )
    fig.suptitle("Matrix using Max LnL SF params", fontsize=10)
    fig.savefig(f"{outdir}/max_lnl_uni.png")

    fig = mock_population.universe.plot_detection_rate_matrix(
        save=False, scatter_events=mock_population.mcz
    )
    fig.suptitle("Matrix using Injected SF params", fontsize=10)
    fig.savefig(f"{outdir}/true_lnl_uni.png")

    horizontal_concat(
        [f"{outdir}/true_lnl_uni.png", f"{outdir}/max_lnl_uni.png"],
        f"{outdir}/uni.png",
        rm_orig=True,
    )

    vertical_concat(
        [f"{outdir}/lnl.png", f"{outdir}/uni.png"],
        f"{outdir}/surface.png",
        rm_orig=False,
    )

    true = dict(muz=mock_population.muz, sigma0=mock_population.sigma0)
    plot1d_fig = plot_1d(large_cache, true)
    plot1d_fig.suptitle("Full range")
    plt.tight_layout()
    plot1d_fig.savefig(f"{outdir}/plot1d_large.png")
    plot1d_fig = plot_1d(zoom_cache, true)
    plot1d_fig.suptitle("Zoom in")
    plt.tight_layout()
    plot1d_fig.savefig(f"{outdir}/plot1d_zoom.png")
    horizontal_concat(
        [f"{outdir}/plot1d_large.png", f"{outdir}/plot1d_zoom.png"],
        f"{outdir}/plot1d.png",
        rm_orig=True,
    )


for idx in [500, 450, 550]:
    main_plotter("grid_data.h5", "focused_data2.h5", idx)
