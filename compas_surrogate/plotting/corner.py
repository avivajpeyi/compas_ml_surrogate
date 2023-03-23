from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from corner import corner


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
) -> plt.Figure:
    """Plot corner plot weighted by the probability of the samples."""
    kwgs = dict(
        smooth=0.9,
        label_kwargs=dict(fontsize=30),
        title_kwargs=dict(fontsize=16),
        color="tab:blue",
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
    )
    if show_datapoints:
        kwgs.update(
            dict(
                plot_datapoints=True,
                plot_contours=True,
                fill_contours=False,
                no_fill_contours=True,
            )
        )

    if labels is None:
        labels = list(samples.keys())
    kwgs["labels"] = labels
    if prob is not None:
        kwgs["weights"] = prob
    if true_params is not None:
        kwgs["truths"] = true_params
    fig = corner(samples, **kwgs)
    return fig
