import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from matplotlib.colors import ListedColormap, to_hex, to_rgba
from scipy.interpolate import Rbf, griddata

sns.set_theme(style="ticks")

GREEN = "#70B375"
ORANGE = "#B37570"
PURPLE = "#7570B3"

SIGMA_LEVELS = [1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.0)]


def make_colormap_to_white(color="tab:orange"):
    color_rgb = np.array(to_rgba(color))
    lower = np.ones((int(256 / 4), 4))
    for i in range(3):
        lower[:, i] = np.linspace(1, color_rgb[i], lower.shape[0])
    cmap = np.vstack(lower)
    return ListedColormap(cmap, name="myColorMap", N=cmap.shape[0])


CMAP = make_colormap_to_white()


def plot_probs(
    x,
    y,
    p,
    xlabel="",
    ylabel="",
    fname=None,
    levels=SIGMA_LEVELS,
    cmap=CMAP,
    true_values=None,
    zoom_in=False,
    zoom_range=None,
):
    plt.close("all")

    fig, axes = plt.subplots(1, 1, figsize=(4, 4))
    ax = axes

    if zoom_range is not None:
        zoom_in = True

    if true_values and zoom_in:

        if zoom_range is None:
            xrange = max(x) - min(x)
            yrange = max(y) - min(y)

            # center around true values
            x_min = true_values[0] - 0.2 * xrange
            x_max = true_values[0] + 0.2 * xrange
            y_min = true_values[1] - 0.2 * yrange
            y_max = true_values[1] + 0.2 * yrange
        else:
            x_min, x_max, y_min, y_max = zoom_range

        # filter data to region
        mask = (x > x_min) & (x < x_max) & (y > y_min) & (y < y_max)
        x = x[mask]
        y = y[mask]
        p = p[mask]

    if isinstance(p, pd.Series):
        z = p.values
    else:
        z = p.copy()
    z[z == -np.inf] = np.nan

    gridx, gridy, gridz = array_to_meshgrid(x, y, z)
    ax.pcolor(
        gridx,
        gridy,
        gridz,
        cmap=cmap,
        vmin=np.nanmin(z),
        vmax=np.nanmax(z),
        zorder=-100,
    )

    ax.tricontour(
        x,
        y,
        z,
        levels,
        vmin=np.nanmin(z),
        vmax=np.nanmax(z),
        colors="k",
        linewidths=0.2,
        alpha=0.5,
    )

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_aspect(1.0 / ax.get_data_ratio())

    if true_values:
        ax.scatter(
            true_values[0], true_values[1], marker="+", s=100, color="k", zorder=100
        )
        ax.vlines(true_values[0], min(y), max(y), color="k", lw=0.5, zorder=100)
        ax.hlines(true_values[1], min(x), max(x), color="k", lw=0.5, zorder=100)

    fig.tight_layout()
    if fname:
        fig.tight_layout()
        fig.savefig(fname)
    else:
        return fig, axes


def get_alpha_colormap(hex_color, level=SIGMA_LEVELS):
    rbga = to_rgba(hex_color)
    return (to_hex((rbga[0], rbga[1], rbga[2], l), True) for l in level)


def array_to_meshgrid(x, y, z, method="cubic", resolution=50):
    gridx, gridy = np.mgrid[
        min(x) : max(x) : complex(resolution), min(y) : max(y) : complex(resolution)
    ]
    points = [(xi, yi) for xi, yi in zip(x, y)]
    gridz = griddata(points, z, (gridx, gridy), method=method)

    return gridx, gridy, gridz
