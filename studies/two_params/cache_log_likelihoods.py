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


def plot_2d_density(x, y, z, true_x=None, true_y=None, scatter=False):
    """
    Plot a 2D density plot
    """
    fig, ax = plt.subplots()
    if scatter:
        cmap = ax.scatter(x, y, c=z, cmap="viridis", s=1)
    else:
        cmap = ax.tricontourf(x, y, z, 100, cmap="viridis")

    if true_x is not None and true_y is not None:
        ax.scatter(true_x, true_y, marker="x", color="red", s=100)
        ax.axvline(true_x, color="red", alpha=0.1)
        ax.axhline(true_y, color="red", alpha=0.1)

    ax.set_aspect("equal")
    ax.set_xlabel("muz")
    ax.set_ylabel("sigma0")
    fig.colorbar(cmap, label="LnL")
    fig.tight_layout()
    fig.savefig("lnl_2d_density" + "_scatter" * scatter + ".png")


def get_params_from_universe_paths(path):
    """
    Extract the parameters from the universe path.
    """
    import regex as re

    param_names = ["n", "aSF", "bSF", "cSF", "dSF", "muz", "", "sigma0"]
    params = re.findall(r"[-+]?\d*\.\d+|\d+", path)
    param_vals = {name: float(param) for name, param in zip(param_names, params)}
    param_vals.pop("")
    return param_vals


def main(universes_glob=GLOB_STR, cache_lnl_file=CACHE_LNL_FILE):
    universe_paths = glob(universes_glob)
    uni_path = random.choice(universe_paths)
    observed_uni = Universe.from_npz(uni_path)
    mock_population = observed_uni.sample_possible_event_matrix()
    mock_population.plot(save=True)
    compute_and_cache_lnl(
        mock_population, cache_lnl_file, universe_paths=universe_paths
    )

    data_dict = load_lnl_cache("cache_lnl.npz")
    plt_kwgs = dict(
        x=data_dict["muz"],
        y=data_dict["sigma0"],
        z=data_dict["lnl"],
        true_x=data_dict["true_params"]["muz"],
        true_y=data_dict["true_params"]["sigma0"],
    )
    plot_2d_density(**plt_kwgs)
    plot_2d_density(**plt_kwgs, scatter=True)


if __name__ == "__main__":
    main()
