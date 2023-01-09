import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

from compas_surrogate.cosmic_integration.universe import (
    Universe,
    plot_event_matrix_on_universe_detection_matrix,
)
from compas_surrogate.liklelihood import ln_likelihood

OUTDIR = "out_universe"
GLOB_STR = f"{OUTDIR}/*.npz"


def set_color(x):
    return ["C0" if xi != max(x) else "C1" for xi in x]


def load_universes(glob_str=GLOB_STR):
    universe_fns = glob(glob_str)
    if len(universe_fns) == 0:
        raise ValueError(
            "No universes found. Run make_detection_matricies.py first."
        )
    universes = [Universe.from_npz(fn) for fn in universe_fns]
    return universes


def plot_lnl(sf_params, lnl, poisson_lnl, mcz_lnl, observed_uni):
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].scatter(
        sf_params,
        poisson_lnl,
        label="COMPAS runs",
        color=set_color(poisson_lnl),
    )
    ax[0].axvline(observed_uni.SF[3], c="C2", label="True SF")
    ax[0].legend(frameon=False)
    ax[0].set_ylabel("Poisson LnL")
    ax[1].scatter(sf_params, mcz_lnl, color=set_color(mcz_lnl))
    ax[1].axvline(observed_uni.SF[3], c="C2")
    ax[1].set_ylabel("lnP (mc,z | SF)")
    ax[2].scatter(sf_params, lnl, color=set_color(lnl))
    ax[2].axvline(observed_uni.SF[3], c="C2")
    ax[2].set_ylabel("lnP + Poisson LnL")
    ax[2].set_xlabel("SF")
    plt.tight_layout()
    return fig


def main():
    universes = load_universes()
    print(f"Loaded {len(universes)} universes")

    # load a set of universes and choose a "true" universe
    observed_uni = universes[5]
    true_events, true_mcz = observed_uni.sample_possible_event_matrix()
    fig = plot_event_matrix_on_universe_detection_matrix(
        observed_uni, true_rate2d=true_events
    )
    fig.savefig(os.path.join(OUTDIR, "true_events.png"))

    lnl_list = np.array(
        [
            ln_likelihood(
                mcz_obs=true_mcz,
                model_prob_func=uni.prob_of_mcz,
                n_model=uni.n_detections(),
                detailed=True,
            )
            for uni in universes
        ]
    )

    sf = [uni.SF[3] for uni in universes]
    fig = plot_lnl(
        sf_params=sf,
        lnl=lnl_list[:, 0],
        poisson_lnl=lnl_list[:, 1],
        mcz_lnl=lnl_list[:, 2],
        observed_uni=observed_uni,
    )
    plt_fname = os.path.join(OUTDIR, "lnl.png")
    fig.savefig(plt_fname)

    # save a npz file with the lnl values
    fname = os.path.join(OUTDIR, "lnl.npz")
    np.savez(
        fname,
        sf=sf,
        lnl=lnl_list[:, 0],
    )

    print(f"Saved {fname}, {plt_fname}")


if __name__ == "__main__":
    main()
