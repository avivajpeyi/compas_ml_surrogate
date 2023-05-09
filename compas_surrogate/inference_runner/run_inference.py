"""Module containing helper functins for sampling using the surrogate likelihood"""
import logging
import os
import bilby
from ..cosmic_integration.star_formation_paramters import get_star_formation_prior
from ..data_generation.likelihood_cacher import get_training_lnl_cache
from ..surrogate.surrogate_likelihood import SurrogateLikelihood
from ..surrogate import get_surrogate_model
from .inference_post_proc import make_inference_plots, make_param_table, make_summary_page
from ..logger import logger

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

    # load training data
    if cache_outdir is None:
        cache_outdir = outdir
    data_cache = get_training_lnl_cache(
        outdir=cache_outdir,
        n_samp=n,
        det_matrix_h5=det_matrix_h5,
        universe_id=universe_id,
        clean=clean,
    )

    # load surrogate model
    model = get_surrogate_model(
        os.path.join(outdir, "model"),
        training_data_cache=data_cache,
        clean=clean,
    )
    sampling_params = data_cache.get_varying_param_keys()
    surrogate_lnl = SurrogateLikelihood(model, parameter_keys=sampling_params)
    prior = get_star_formation_prior(sampling_params)

    # run sampler
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
            plots=False,
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

    # make plots
    mock_fname = f"{cache_outdir}/mock_uni.npz"
    make_inference_plots(mock_fname, surr_fname, det_matrix_h5, data_cache, model)
    table = make_param_table(surr_result, data_cache)
    logger.info(f"Inference Summary:\n{table}")
    make_summary_page(outdir, table)
    return surr_result
