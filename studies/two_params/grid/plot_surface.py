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
from compas_surrogate.utils import exp_norm_scale_log_data


def make_plots(det_fname, mock_uni, outdir):
    cache = get_training_lnl_cache(
        outdir=outdir, det_matrix_h5=det_fname, mock_uni=mock_uni, clean=False
    )
    cache_subset = cache.dataframe
    z = cache_subset["lnl"] - cache_subset["lnl"].max()
    x, y = cache_subset["muz"].values, cache_subset["sigma0"].values

    data = pd.DataFrame({"x": x, "y": y, "z": z})
    num_x, num_y = len(np.unique(x)), len(np.unique(y))
    plt_range = [x.min(), x.max(), y.min(), y.max()]

    cmap = "RdBu"
    fig, axes = plt.subplots(3, 1, figsize=(4, 12))
    # scatter size is to cover the whole area
    cbar = axes[0].scatter(
        data.x, data.y, c=data.z, s=1, cmap=cmap, marker="s", vmin=-1, vmax=0
    )
    # add colorbar
    fig.colorbar(cbar, ax=axes[0])

    # tricontourf
    cb = axes[1].tricontourf(data.x, data.y, data.z, 100, cmap=cmap, vmin=-1, vmax=0)
    fig.colorbar(cb, ax=axes[1])

    # imshow x, y, z interpolated data
    xi = np.linspace(data.x.min(), data.x.max(), num_x)
    yi = np.linspace(data.y.min(), data.y.max(), 100)
    zi = griddata((data.x, data.y), data.z, (xi[None, :], yi[:, None]), method="linear")
    cb = axes[2].pcolormesh(xi, yi, zi, cmap=cmap, vmax=0, vmin=-1)
    fig.colorbar(cb, ax=axes[2])
    for ax in axes:
        ax.axis(plt_range)

    plt.tight_layout()
    fig.savefig(f"{outdir}/lnl.png", dpi=300)


for idx in [100, 500, 200, 1000, 666]:
    mock_uni = Universe.from_hdf5("focused_data.h5", idx)
    make_plots("focused_data.h5", mock_uni, f"mock_uni_{idx}")
