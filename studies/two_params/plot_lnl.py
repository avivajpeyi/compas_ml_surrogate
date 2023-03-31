import matplotlib.pyplot as plt
import numpy as np

from compas_surrogate.data_generation.likelihood_cacher import get_training_lnl_cache
from compas_surrogate.plotting.hist2d import plot_probs
from compas_surrogate.utils import exp_norm_scale_log_data

OUT = "lnl_plots"

cache = get_training_lnl_cache(
    outdir=OUT, det_matrix_h5="det_matrix.h5", universe_id=10, clean=False
)

cache_subset = cache.dataframe
norm_p = exp_norm_scale_log_data(cache_subset["lnl"])

LEVELS = np.quantile(norm_p, [0.1, 0.5, 0.68, 0.95, 1])
true_params = [cache.true_dict["muz"], cache.true_dict["sigma0"]]
fig, ax = plot_probs(
    cache_subset["muz"],
    cache_subset["sigma0"],
    norm_p,
    levels=LEVELS,
    cmap="Oranges",
    true_values=true_params,
)
fig.show()

fig, ax = plot_probs(
    cache_subset["muz"],
    cache_subset["sigma0"],
    norm_p,
    levels=LEVELS,
    cmap="Oranges",
    true_values=true_params,
    zoom_in=True,
)
fig.show()
