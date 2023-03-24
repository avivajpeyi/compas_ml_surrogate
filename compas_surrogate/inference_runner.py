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
from .cosmic_integration.universe import Universe
from .data_generation.likelihood_cacher import LikelihoodCache, compute_and_cache_lnl
from .logger import logger
from .surrogate.models import DeepGPModel, SklearnGPModel
from .surrogate.surrogate_likelihood import SurrogateLikelihood


def get_training_lnl_cache(
    outdir, n_samp=None, det_matrix_h5=None, universe_id=None, clean=False
) -> LikelihoodCache:
    """
    Get the likelihood cache --> used for training the surrogate
    Specify the det_matrix_h5 and universe_id to generate a new cache

    :param outdir: outdir to store the cache (stored as OUTDIR/cache_lnl.npz)
    :param n_samp: number of samples to save in the cache (all samples used if None)
    :param det_matrix_h5: the detection matrix used to generate the lnl cache
    :param universe_id: the universe id used to generate the lnl cache
    """
    cache_file = f"{outdir}/cache_lnl.npz"
    if clean and os.path.exists(cache_file):
        logger.info(f"Removing cache {cache_file}")
        os.remove(cache_file)
    if os.path.exists(cache_file):
        logger.info(f"Loading cache from {cache_file}")
        lnl_cache = LikelihoodCache.from_npz(cache_file)
    else:
        os.makedirs(outdir, exist_ok=True)
        h5_file = h5py.File(det_matrix_h5, "r")
        total_n_det_matricies = len(h5_file["detection_matricies"])
        if universe_id is None:
            universe_id = random.randint(0, total_n_det_matricies)
        assert (
            universe_id < total_n_det_matricies
        ), f"Universe id {universe_id} is larger than the number of det matricies {total_n_det_matricies}"
        mock_uni = Universe.from_hdf5(h5_file, universe_id)
        mock_population = mock_uni.sample_possible_event_matrix()
        mock_population.plot(save=True, outdir=outdir)
        logger.info(
            f"Generating cache {cache_file} using {det_matrix_h5} and universe {universe_id}:{mock_population}"
        )
        lnl_cache = compute_and_cache_lnl(
            mock_population, cache_file, h5_path=det_matrix_h5
        )

    plt_fname = cache_file.replace(".npz", ".png")
    lnl_cache.plot(fname=plt_fname, show_datapoints=True)
    if n_samp is not None:
        lnl_cache = lnl_cache.sample(n_samp)
    lnl_cache.plot(
        fname=plt_fname.replace(".png", "_training.png"), show_datapoints=True
    )
    return lnl_cache


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
        in_data = param_data.T
        out_data = training_data_cache.lnl.reshape(-1, 1)
        logger.info(
            f"Training model {model_dir}: IN[{in_data.shape}]--> OUT[{out_data.shape}]"
        )
        model = gp_model()
        metrics = model.train(in_data, out_data, verbose=True, savedir=model_dir)
        logger.info(f"Surrogate metrics: {metrics}")
        pred_lnl = model(training_data_cache.true_param_vals)
        logger.info(
            f"True LnL: {training_data_cache.true_lnl}, Surrogate LnL: {pred_lnl}"
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
            nlive=250,
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
    surr_result.save_to_file(extension="json")
    fig = surr_result.plot_corner(save=False)
    true_lnl = data_cache.true_lnl
    pred_lnl = model.prediction_str(data_cache.true_param_vals)
    fig.suptitle(f"model lnl: {true_lnl:.2f}\n" f"surro lnl: ${pred_lnl}$")
    fig.savefig(os.path.join(outdir, "corner.png"))
    return surr_result
