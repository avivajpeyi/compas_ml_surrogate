from typing import List

import numpy as np
from tqdm.contrib.concurrent import process_map

from compas_surrogate.cosmic_integration.universe import (
    MockPopulation,
    Universe,
)
from compas_surrogate.liklelihood import ln_likelihood
from compas_surrogate.logger import logger
from compas_surrogate.utils import get_num_workers


def _get_lnl_and_param(uni_fn, observed_mcz):
    uni = Universe.from_npz(uni_fn)
    lnl = ln_likelihood(
        mcz_obs=observed_mcz,
        model_prob_func=uni.prob_of_mcz,
        n_model=uni.n_detections(),
        detailed=False,
    )
    params = uni.param_list()
    logger.debug(f"Processed {uni_fn} with lnl={lnl} and params={params}")
    return np.array([lnl, *params])


def compute_and_cache_lnl(
    mock_population: MockPopulation, universe_paths: List, cache_lnl_file: str
):
    n = len(universe_paths)
    logger.info(f"Starting LnL computation for {n} universes")

    lnl_and_param_list = process_map(
        _get_lnl_and_param,
        universe_paths,
        [mock_population.mcz] * n,
        desc="Computing likelihoods",
        max_workers=get_num_workers(),
    )

    lnl_and_param_list = np.array(lnl_and_param_list)
    np.savez(
        cache_lnl_file,
        lnl=lnl_and_param_list[:, 0],
        params=lnl_and_param_list[:, 1:],
    )
    logger.success(f"Saved {cache_lnl_file}")
