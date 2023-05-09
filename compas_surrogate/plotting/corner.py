from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from corner import corner
from matplotlib.lines import Line2D

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

    s_labels = list(samples.keys())
    _s = samples[s_labels[0]]

    bins = int(np.sqrt(len(_s)))

    s_array = np.array([samples[k] for k in s_labels]).T

    if len(samples) != 1:
        fig = corner(s_array, **kwgs, bins=bins, labels=labels)
    else:
        plt.figure(figsize=(3, 3))
        plt.hist(
            s_array.ravel(),
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


def add_legend_to_corner(fig, labels, colors, fs=16):
    """Add legend to corner plot"""
    legend_elements = [
        Line2D([0], [0], color=c, lw=4, label=l) for c, l in zip(colors, labels)
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.95, 0.95),
        bbox_transform=fig.transFigure,
        frameon=False,
        fontsize=fs,
    )
    return fig
