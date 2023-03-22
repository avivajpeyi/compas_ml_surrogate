from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .corner import plot_corner


def plot_1d_lnl(
        params: List[float],
        lnl: List[float],
        poisson_lnl: List[float] = [],
        mcz_lnl: List[float] = [],
        true_param: Optional[float] = None,
):
    """Plot 1d Lnl for a given parameter."""
    set_color = lambda x: ["C0" if xi != max(x) else "C1" for xi in x]

    n_plots = 1 + len(poisson_lnl) + len(mcz_lnl)
    fig, ax = plt.subplots(n_plots, 1, sharex=True)
    ax[0].scatter(
        params,
        lnl,
        label="COMPAS runs",
        color=set_color(poisson_lnl),
    )
    ax[0].set_ylabel("LnL")
    if len(poisson_lnl) > 0:
        ax[1].scatter(
            params,
            poisson_lnl,
            color=set_color(poisson_lnl),
        )
        ax[1].set_ylabel("Poisson LnL")
    if len(mcz_lnl) > 0:
        ax[2].scatter(
            params,
            mcz_lnl,
            color=set_color(mcz_lnl),
        )
        ax[2].set_ylabel("MCZ LnL")

    if true_param is not None:
        for axi in ax:
            axi.axvline(true_param, color="C1", label="True value")
    ax[0].legend(frameon=False)
    ax[0].set_xlabel("SF")
    plt.tight_layout()
    return fig


def plot_hacky_1d_lnl(
        params: pd.DataFrame,
        lnl: np.ndarray,
        true_param: Dict[str, float] = {},
        plt_kwgs={}
):
    # fig with len(params) subplots
    nparams = len(params.columns)
    fig, ax = plt.subplots(nparams, 1, figsize=(4, 2 * nparams))

    for i, param in enumerate(params.columns):
        p = params[param]
        min_p, max_p = min(p), max(p)
        nbins = max(1, int(len(p) / 200))
        bins = np.linspace(min_p, max_p, nbins)
        binx = (max_p - min_p) / nbins
        lnl_in_bins = [lnl[(p > b) & (p < b + binx)] for b in bins]
        bins = [(b1 + b2) / 2 for b1, b2 in zip(bins[:-1], bins[1:])]
        lnl_in_bins = lnl_in_bins[:-1]
        counts = [len(l) for l in lnl_in_bins]
        p_lnl = [np.trapz(l) / c for l, c in zip(lnl_in_bins, counts)]
        unc_lnl = [np.std(l) for l, c in zip(lnl_in_bins, counts)]
        plt_kwgs['fmt'] = plt_kwgs.get('fmt', 'o')
        ax[i].errorbar(bins, p_lnl, yerr=unc_lnl, **plt_kwgs)
        ax[i].set_xlabel(param)
        if param in true_param:
            ax[i].axvline(true_param[param], color="tab:orange", label="True value")
    plt.tight_layout()
    return fig
