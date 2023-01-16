import random
from glob import glob

import matplotlib.pyplot as plt

from compas_surrogate.cosmic_integration.universe import Universe
from compas_surrogate.data_generation.likelihood_cacher import (
    compute_and_cache_lnl,
    load_lnl_cache,
)

OUTDIR = "../out_muz_sigma0"
GLOB_STR = f"{OUTDIR}/*.npz"

CACHE_LNL_FILE = "cache_lnl.npz"


def plot_2d_density(x, y, z, true_x=None, true_y=None):
    """
    Plot a 2D density plot
    """
    fig, ax = plt.subplots()
    cmap = ax.tricontourf(x, y, z, 100, cmap="viridis")
    if true_x is not None and true_y is not None:
        ax.scatter(true_x, true_y, marker="x", color="red", s=100)
        ax.axvline(true_x, color="red")
        ax.axhline(true_y, color="red")
    ax.set_xlabel("muz")
    ax.set_ylabel("sigma0")
    fig.colorbar(cmap, label="LnL")
    fig.savefig("lnl_2d_density.png")


def main(universes_glob=GLOB_STR, cache_lnl_file=CACHE_LNL_FILE):
    universe_paths = glob(universes_glob)
    random_uni = random.choice(universe_paths)
    observed_uni = Universe.from_npz(random_uni)
    mock_population = observed_uni.sample_possible_event_matrix()
    compute_and_cache_lnl(mock_population, universe_paths, cache_lnl_file)

    data_dict = load_lnl_cache("cache_lnl.npz")
    plot_2d_density(
        data_dict["muz"],
        data_dict["sigma0"],
        data_dict["lnl"],
        true_x=data_dict["true_params"]["muz"],
        true_y=data_dict["true_params"]["sigma0"],
    )


if __name__ == "__main__":
    main()
