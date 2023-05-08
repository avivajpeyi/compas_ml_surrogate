from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from corner import corner

KWGS = dict(
    smooth=0.9,
    label_kwargs=dict(fontsize=30),
    title_kwargs=dict(fontsize=16),
    truth_color="tab:orange",
    quantiles=[0.16, 0.84],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9.0 / 2.0)),
    plot_density=False,
    plot_datapoints=False,
    plot_contours=True,
    fill_contours=True,
    no_fill_contours=False,
    max_n_ticks=3,
    verbose=False,
    use_math_text=True,
    data_kwargs=dict(alpha=0.75),
)


def _clean_samples(samples: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """Remove samples with nan values or  if all values are the same."""
    print(samples)
    samples = {k: v for k, v in samples.items() if not np.isnan(v).any()}
    samples = {k: v for k, v in samples.items() if len(set(v)) > 1}
    return samples


def plot_corner(
    samples: Dict[str, List[float]],
    prob: Optional[Union[List[float], np.ndarray]] = None,
    true_params: Optional[List[float]] = None,
    labels=None,
    show_datapoints=False,
    color="tab:blue",
) -> plt.Figure:
    """Plot corner plot weighted by the probability of the samples."""
    kwgs = KWGS.copy()
    if show_datapoints:
        kwgs.update(
            dict(
                plot_datapoints=True,
                plot_contours=True,
                fill_contours=False,
                no_fill_contours=True,
            )
        )
    kwgs["color"] = color

    if labels is None:
        labels = list(samples.keys())
    kwgs["labels"] = labels
    if prob is not None:
        kwgs["weights"] = prob
    if true_params is not None:
        kwgs["truths"] = true_params

    _s = samples[list(samples.keys())[0]]

    if len(_s) > 20000:
        bins = 50
    elif len(_s) > 1000:
        bins = 20
    else:
        bins = 10

    if len(samples) != 1:
        fig = corner(samples, **kwgs, bins=bins)
    else:
        plt.figure(figsize=(3, 3))
        plt.hist(
            samples[list(samples.keys())[0]],
            weights=prob,
            bins=bins,
            histtype="step",
            color=color,
            density=True,
        )
        plt.xlabel(labels[0])
        if true_params is not None:
            plt.axvline(true_params[0], color="tab:orange", label="True value")
        plt.tight_layout()
        fig = plt.gcf()
    return fig
