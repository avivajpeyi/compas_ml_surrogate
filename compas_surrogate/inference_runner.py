"""Module containing helper functins for sampling using the surrogate likelihood"""
import os
import random
import shutil
from glob import glob
from typing import List, Optional, Union

import bilby
import h5py
import matplotlib.pyplot as plt
import numpy as np

from .cosmic_integration.star_formation_paramters import get_star_formation_prior
from .cosmic_integration.universe import MockPopulation, Universe
from .data_generation.detection_matrix_generator import (
    get_universe_closest_to_parameters,
)
from .data_generation.likelihood_cacher import LikelihoodCache, get_training_lnl_cache
from .logger import logger
from .plotting.image_utils import horizontal_concat
from .surrogate.models import DeepGPModel, SklearnGPModel
from .surrogate.surrogate_likelihood import SurrogateLikelihood


def get_ml_surrogate_model(
    model_dir: str,
    training_data_cache: Optional[LikelihoodCache] = None,
    gp_model=SklearnGPModel,
    clean=False,
):
    """
    Get the ML surrogate model
    """
    if clean and os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    if os.path.exists(model_dir):
        logger.info(f"Loading model from {model_dir}")
        model = gp_model.load(model_dir)
    else:
        param_data = training_data_cache.get_varying_params()
        tru_lnl = training_data_cache.true_lnl

        # ensure that the true lnl is a valid value
        if not np.isfinite(tru_lnl):
            raise ValueError("True lnl is not finite! Skipping analysis.")

        in_data = param_data.T
        out_data = training_data_cache.lnl.reshape(-1, 1)
        logger.info(
            f"Training model {model_dir}: IN[{in_data.shape}]--> OUT[{out_data.shape}]"
        )
        model = gp_model()
        plt_kwgs = dict(
            labels=training_data_cache.get_varying_param_keys(),
            truths=training_data_cache.true_param_vals.ravel(),
        )

        metrics = model.train(
            in_data, out_data, verbose=True, savedir=model_dir, extra_kwgs=plt_kwgs
        )
        logger.info(f"Surrogate metrics: {metrics}")

        # check if true lnl inside the range of pred_lnl
        pred_lnl = np.array(model(training_data_cache.true_param_vals)).ravel()
        check = (pred_lnl[0] <= tru_lnl) & (tru_lnl <= pred_lnl[2])
        diff = np.max(
            [np.abs(pred_lnl[1] - pred_lnl[0]), np.abs(pred_lnl[1] - pred_lnl[2])]
        )
        pred_str = f"{pred_lnl[1]:.2f} +/- {diff:.2f}"
        check_str = "✔" * 3 if check else "❌" * 3
        logger.info(
            f"{check_str} True LnL: {tru_lnl:.2f}, Surrogate LnL: {pred_str} {check_str}"
        )
    logger.success("Trained and saved Model")

    return model


def run_inference(
    outdir,
    det_matrix_h5=None,
    universe_id=None,
    cache_outdir=None,
    n=None,
    clean=False,
    sampler="dynesty",
) -> bilby.core.result.Result:
    os.makedirs(outdir, exist_ok=True)

    if cache_outdir is None:
        cache_outdir = outdir
    data_cache = get_training_lnl_cache(
        outdir=cache_outdir,
        n_samp=n,
        det_matrix_h5=det_matrix_h5,
        universe_id=universe_id,
        clean=clean,
    )

    model = get_ml_surrogate_model(
        os.path.join(outdir, "model"),
        training_data_cache=data_cache,
        clean=clean,
    )
    sampling_params = data_cache.get_varying_param_keys()
    surrogate_lnl = SurrogateLikelihood(model, parameter_keys=sampling_params)
    prior = get_star_formation_prior(sampling_params)

    if sampler == "emcee":
        smplr_kwargs = dict(
            sampler="emcee",
            nwalkers=100,
            nsteps=1000,
        )
    elif sampler == "dynesty":
        smplr_kwargs = dict(
            sampler="dynesty",
            nlive=1000,
        )

    surr_result = bilby.run_sampler(
        likelihood=surrogate_lnl,
        priors=prior,
        injection_parameters=data_cache.true_dict,
        outdir=outdir,
        label=f"surr_run_{n}",
        clean=clean,
        **smplr_kwargs,
    )
    surr_fname = f"{outdir}/inference_result.json"
    surr_result.save_to_file(filename=surr_fname)
    mock_fname = f"{cache_outdir}/mock_uni.npz"
    make_inference_plots(mock_fname, surr_fname, det_matrix_h5, data_cache, model)
    return surr_result


def make_inference_plots(mock_npz, inference_json, det_matrix_h5, data_cache, model):
    """
    Make a comparison plot of the true  likelihood
    """

    # load data
    mock_uni = MockPopulation.from_npz(mock_npz)
    inference_result = bilby.result.read_in_result(inference_json)
    outdir = os.path.dirname(inference_json)

    # get the inferred universe -- highest likelihood universe
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

    # plot the sampling result
    fig = inference_result.plot_corner(save=False, bins=30, priors=True)
    true_lnl = data_cache.true_lnl
    pred_lnl = model.prediction_str(data_cache.true_param_vals)
    fig.suptitle(f"model lnl: {true_lnl:.2f}\n" f"surro lnl: ${pred_lnl}$")
    corner_plt = f"{outdir}/corner.png"
    fig.savefig(corner_plt)

    # horizontal concatenation
    horizontal_concat(
        [mock_plt, inferred_plt, corner_plt], f"{outdir}/sampling_summary.png"
    )
