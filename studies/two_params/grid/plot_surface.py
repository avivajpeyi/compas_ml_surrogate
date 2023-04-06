import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import griddata

from compas_surrogate.data_generation.likelihood_cacher import (
    Universe,
    get_training_lnl_cache,
)
from compas_surrogate.plotting.hist2d import plot_probs
from compas_surrogate.plotting.image_utils import horizontal_concat
from compas_surrogate.utils import exp_norm_scale_log_data


def make_plots(det_fname, mock_uni, outdir, label=""):
    cache = get_training_lnl_cache(
        outdir=outdir, det_matrix_h5=det_fname, mock_uni=mock_uni, clean=True
    )
    cache_subset = cache.dataframe
    z = cache_subset["lnl"] - cache_subset["lnl"].max()
    x, y = cache_subset["muz"].values, cache_subset["sigma0"].values
    data = pd.DataFrame({"x": x, "y": y, "z": z})
    num_x, num_y = len(np.unique(x)), len(np.unique(y))
    plt_range = [x.min(), x.max(), y.min(), y.max()]
    true_x, true_y = mock_uni.muz, mock_uni.sigma0

    cmap_args = dict(cmap="RdBu", vmin=-3, vmax=0)
    fig, axes = plt.subplots(2, 1, figsize=(5, 8))
    # scatter size is to cover the whole area
    cbar = axes[0].scatter(data.x, data.y, c=data.z, s=1, marker="s", **cmap_args)
    # add colorbar horizontally
    fig.colorbar(cbar, ax=axes[0])

    # imshow x, y, z interpolated data
    xi = np.linspace(data.x.min(), data.x.max(), num_x)
    yi = np.linspace(data.y.min(), data.y.max(), num_y)
    zi = griddata((data.x, data.y), data.z, (xi[None, :], yi[:, None]), method="linear")
    cb = axes[1].pcolormesh(xi, yi, zi, **cmap_args)
    fig.colorbar(cb, ax=axes[1])
    for ax in axes:
        ax.axis(plt_range)
        ax.set_xlabel("muz")
        ax.set_ylabel("sigma0")
        ax.scatter(true_x, true_y, marker="+", color="k", s=100)
        ax.axvline(true_x, color="k", linestyle="--", lw=0.5)
        ax.axhline(true_y, color="k", linestyle="--", lw=0.5)

    fig.suptitle(f"{label}")

    plt.tight_layout()
    fig.savefig(f"{outdir}/lnl_{label}.png", dpi=300)


for idx in [500]:  # , 450, 550]:
    mock_uni = Universe.from_hdf5("focused_data2.h5", idx)
    make_plots("focused_data.h5", mock_uni, f"mock_uni_{idx}", label="zoom_1")
    make_plots("focused_data2.h5", mock_uni, f"mock_uni_{idx}", label="zoom_2")
    horizontal_concat(
        [f"mock_uni_{idx}/lnl_zoom_1.png", f"mock_uni_{idx}/lnl_zoom_2.png"],
        f"mock_uni_{idx}/lnl.png",
    )
