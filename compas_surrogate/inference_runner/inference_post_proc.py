import os

import bilby
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF

from ..cosmic_integration.universe import MockPopulation
from ..data_generation.detection_matrix_generator import (
    get_universe_closest_to_parameters,
)
from ..data_generation.likelihood_cacher import LikelihoodCache
from ..plotting.corner import add_legend_to_corner
from ..plotting.image_utils import horizontal_concat
from ..surrogate.models.model import Model
from ..utils import fmt_val_upper_lower


def make_param_table(
    inference_result: bilby.result.Result, training_cache: LikelihoodCache = None
) -> pd.DataFrame:
    """Make a table of the parameters and their values from the inference run"""
    params = inference_result.search_parameter_keys
    # get median values
    post = inference_result.posterior
    med_strs = []
    for key in params:
        m = np.median(post[key])
        q = np.quantile(post[key], [0.16, 0.84])
        t, b = max(np.abs(q - m)), min(np.abs(q - m))
        m, t, b = round(m, 2), round(t, 2), round(b, 2)
        # if t almost 0
        if t < 0.01 or b < 0.01:
            med_strs.append(f"{m:.2f}")
        else:
            med_strs.append(fmt_val_upper_lower(m, t, b))
    med_strs += [f"-"]

    max_lnl_idx = np.argmax(post.log_likelihood)
    max_lnl_params = post.iloc[max_lnl_idx].to_dict()
    maxl_strs = [f"{max_lnl_params[key]:.2f}" for key in params]
    maxl_strs += [f"{post.log_likelihood.max():.2f}"]

    vals = [med_strs, maxl_strs]
    index = ["median", "maxlnl"]

    if training_cache is not None:
        inj = training_cache.true_dict
        inj_strs = [f"{inj[key]:.2f}" for key in params]
        inj_strs += [f"{training_cache.get_true_param_lnl():.2f}"]
        vals.append(inj_strs)
        index.append("inj")

    params = [*params, "lnl"]
    df = pd.DataFrame(vals, columns=params, index=index)
    return df.T


def make_inference_plots(
    mock_npz: str,
    inference_json: str,
    det_matrix_h5: str,
    data_cache: LikelihoodCache,
    model: Model,
):
    """Make plots for the inference run"""
    inference_result = bilby.result.read_in_result(inference_json)
    outdir = os.path.dirname(inference_json)
    uni_fnames = plot_universes(inference_result, det_matrix_h5, outdir, mock_npz)
    corner_fname = plot_sampling_corner(inference_result, data_cache, model, outdir)
    fnames = [*uni_fnames, corner_fname]
    horizontal_concat(fnames, f"{outdir}/sampling_summary.png", rm_orig=False)


def plot_universes(
    inference_result: bilby.core.result.Result,
    det_matrix_h5: str,
    outdir: str,
    mock_npz: str,
):
    # get the inferred universe -- highest likelihood universe
    mock_uni = MockPopulation.from_npz(mock_npz)
    max_lnl_idx = np.argmax(inference_result.posterior.log_likelihood)
    max_lnl_params = inference_result.posterior.iloc[max_lnl_idx].to_dict()
    inferred_uni = get_universe_closest_to_parameters(
        det_matrix_h5, list(max_lnl_params.values())
    )

    # plot the 'true' universe
    outdir_mock = os.path.dirname(mock_npz)
    mock_plt = f"{outdir_mock}/injection.png"
    if not os.path.exists(mock_plt):
        mock_uni.plot(fname=mock_plt)

    # plot the 'inferred' universe
    fig = inferred_uni.plot_detection_rate_matrix(
        save=False, scatter_events=mock_uni.mcz
    )
    fig.suptitle("MaxLnL Inferred Universe")
    inferred_plt = f"{outdir}/inferred.png"
    fig.savefig(inferred_plt)
    plt.close(fig)
    return mock_plt, inferred_plt


def plot_sampling_corner(
    inference_result: bilby.core.result.Result,
    data_cache: LikelihoodCache,
    model: Model,
    outdir: str,
):
    bins_size = int(np.sqrt(len(inference_result.posterior)) * 0.5)
    plot_parameters = list(inference_result.search_parameter_keys)
    fig = inference_result.plot_corner(save=False, bins=bins_size)
    bins = {
        p: np.histogram(inference_result.posterior[p], bins=bins_size)[1]
        for p in plot_parameters
    }
    train_hists = data_cache.get_sample_histograms(bins)
    axes = fig.get_axes()
    for i, par in enumerate(plot_parameters):
        ax = axes[i + i * len(plot_parameters)]
        ytop = ax.get_ylim()[1] * 0.95
        hist_data = train_hists[par]
        be = hist_data.bin_edges
        h = hist_data.hist
        ax.hist(be[:-1], be, weights=h / np.max(h) * ytop, color="C2", histtype="step")
    true_lnl = data_cache.true_lnl
    pred_lnl = model.prediction_str(data_cache.true_param_vals)
    fig.suptitle(f"model lnl: {true_lnl:.2f}\n" f"surro lnl: ${pred_lnl}$")
    corner_plt = f"{outdir}/corner.png"
    fig = add_legend_to_corner(
        fig, labels=["Inferred", "Train", "True"], colors=["C0", "C2", "tab:orange"]
    )
    fig.savefig(corner_plt)
    plt.close(fig)
    return corner_plt


def make_summary_page(outdir, table, cell_height=8):
    """Make a summary page for the inference run"""
    ch = cell_height
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font(
        "Arial",
        size=12,
    )
    pdf.cell(200, 10, txt="Inference Summary", ln=1, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    # Table Header
    pdf.set_font("Arial", "B", 10)
    for i in range(0, len(table.columns)):
        pdf.cell(w=40, h=ch, txt=table.columns[i], border=1, ln=i, align="C")
    # Table contents
    pdf.set_font("Arial", "", 10)
    for i in range(0, len(table)):
        for col in range(0, len(table.columns)):
            col_name = table.columns[col]
            pdf.cell(
                w=40, h=ch, txt=table[col_name].iloc[i], border=1, ln=col, align="C"
            )

    # TODO: add plots

    pdf.output(f"{outdir}/summary.pdf")
