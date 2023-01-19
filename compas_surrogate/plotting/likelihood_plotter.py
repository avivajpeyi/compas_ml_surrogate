from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from corner import corner


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
