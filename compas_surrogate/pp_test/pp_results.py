import glob
from collections import namedtuple
from itertools import product
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import tqdm
from bilby.core.result import Result

from ..logger import logger


class PPresults:
    def __init__(self, ci: pd.DataFrame):
        self.ci = ci

    def to_csv(self, csv):
        """Save all results to csv"""
        self.ci.to_csv(csv, index=False)

    @classmethod
    def from_csv(cls, csv):
        """Load all results from csv"""
        return cls(pd.read_csv(csv))

    @classmethod
    def from_csvs(cls, samples_regex, injection_csv):
        """Load all results from csv"""
        cred_int = []
        injections = pd.read_csv(injection_csv)
        for f in tqdm.tqdm(glob.glob(samples_regex)):
            post = pd.read_csv(f)
            ci = {}
            # get injection parameters for this sample
            injection_parameters = (
                {}
            )  # TODO: need a way to match the injection parameters to the posteior
            for p in post.columns:
                if p not in injection_parameters:
                    continue
                ci[p] = sum(np.array(post[p] < injection_parameters[p]) / len(post))
            cred_int.append(ci)
        return cls(pd.DataFrame(cred_int))

    @classmethod
    def from_results(cls, regex):
        """Load all results from regex"""
        cred_int = []
        for f in tqdm.tqdm(glob.glob(regex)):
            r = Result.from_json(f)
            ci = {}
            for p in r.posterior.columns:
                if p not in r.injection_parameters:
                    continue
                post = r.posterior[p]
                ci[p] = sum(np.array(post < r.injection_parameters[p]) / len(post))
            cred_int.append(ci)
        return cls(pd.DataFrame(cred_int))

    def plot(self, fname="pp_plot.png"):
        fig, pvals = plot_credible_intervals(self.ci)
        fig.savefig(fname)
        plt.close(fig)
        return fig, pvals


def plot_credible_intervals(
    credible_levels: pd.DataFrame, confidence_interval=[0.68, 0.95, 0.997]
) -> Tuple[plt.Figure, namedtuple]:
    """Plot credible intervals for a set of parameters"""
    fig, ax = plt.subplots()
    colors = ["C{}".format(i) for i in range(8)]
    linestyles = ["-", "--", ":"]
    lines = ["{}{}".format(a, b) for a, b in product(linestyles, colors)]
    x_values = np.linspace(0, 1, 1001)
    N = len(credible_levels)
    confidence_interval_alpha = [0.1] * len(confidence_interval)
    for ci, alpha in zip(confidence_interval, confidence_interval_alpha):
        edge_of_bound = (1.0 - ci) / 2.0
        lower = scipy.stats.binom.ppf(1 - edge_of_bound, N, x_values) / N
        upper = scipy.stats.binom.ppf(edge_of_bound, N, x_values) / N
        # The binomial point percent function doesn't always return 0 @ 0,
        # so set those bounds explicitly to be sure
        lower[0] = 0
        upper[0] = 0
        ax.fill_between(x_values, lower, upper, alpha=alpha, color="k")
    pvalues = []
    logger.info("Key: KS-test p-value")
    for ii, key in enumerate(credible_levels):
        pp = np.array(
            [
                sum(credible_levels[key].values < xx) / len(credible_levels)
                for xx in x_values
            ]
        )
        pvalue = scipy.stats.kstest(credible_levels[key], "uniform").pvalue
        pvalues.append(pvalue)
        logger.info("{}: {}".format(key, pvalue))
        label = "{} ({:2.3f})".format(key, pvalue)
        plt.plot(x_values, pp, lines[ii], label=label)
    Pvals = namedtuple("pvals", ["combined_pvalue", "pvalues", "names"])
    pvals = Pvals(
        combined_pvalue=scipy.stats.combine_pvalues(pvalues)[1],
        pvalues=pvalues,
        names=list(credible_levels.keys()),
    )
    logger.info("Combined p-value: {}".format(pvals.combined_pvalue))
    ax.set_title(
        "N={}, p-value={:2.4f}".format(len(credible_levels), pvals.combined_pvalue)
    )
    ax.set_xlabel("C.I.")
    ax.set_ylabel("Fraction of events in C.I.")
    # ax legend to the right of the plot
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # keep the x and y axis the same size (square)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    return fig, pvals
