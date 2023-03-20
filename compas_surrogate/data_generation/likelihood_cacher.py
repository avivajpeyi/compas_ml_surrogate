from itertools import repeat
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from compas_surrogate.cosmic_integration.universe import (
    MockPopulation,
    Universe,
)
from compas_surrogate.liklelihood import LikelihoodCache, ln_likelihood
from compas_surrogate.logger import logger
from compas_surrogate.utils import get_num_workers


def _get_lnl_and_param_uni(uni: Universe, observed_mcz: np.ndarray):
    lnl = ln_likelihood(
        mcz_obs=observed_mcz,
        model_prob_func=uni.prob_of_mcz,
        n_model=uni.n_detections(),
        detailed=False,
    )
    logger.debug(f"Processed {uni} lnl={lnl}.")
    return np.array([lnl, *uni.param_list])


def _get_lnl_and_param_from_npz(npz_fn: str, observed_mcz: np.ndarray):
    uni = Universe.from_npz(npz_fn)
    return _get_lnl_and_param_uni(uni, observed_mcz)


def _get_lnl_and_param_from_h5(
    h5_path: h5py.File, idx: int, observed_mcz: np.ndarray
):
    uni = Universe.from_hdf5(h5py.File(h5_path, "r"), idx)
    return _get_lnl_and_param_uni(uni, observed_mcz)


def compute_and_cache_lnl(
    mock_population: MockPopulation,
    cache_lnl_file: str,
    h5_path: Optional[str] = "",
    universe_paths: Optional[List] = None,
):
    if universe_paths is not None:
        n = len(universe_paths)
        args = (
            _get_lnl_and_param_from_npz,
            universe_paths,
            repeat(mock_population.mcz),
        )
    elif h5_path is not None:
        n = len(h5py.File(h5_path, "r")["parameters"])
        args = (
            _get_lnl_and_param_from_h5,
            repeat(h5_path),
            range(n),
            repeat(mock_population.mcz),
        )
    else:
        raise ValueError("Must provide either hf5_path or universe_paths")

    logger.info(f"Starting LnL computation for {n} universes")

    lnl_and_param_list = np.array(
        process_map(
            *args,
            desc="Computing likelihoods",
            max_workers=get_num_workers(),
            chunksize=100,
            total=n,
        )
    )
    true_lnl = ln_likelihood(
        mcz_obs=mock_population.mcz,
        model_prob_func=mock_population.universe.prob_of_mcz,
        n_model=mock_population.universe.n_detections(),
    )
    lnl_cache = LikelihoodCache(
        lnl=lnl_and_param_list[:, 0],
        params=lnl_and_param_list[:, 1:],
        true_params=mock_population.param_list,
        true_lnl=true_lnl,
    )
    lnl_cache.save(cache_lnl_file)
    logger.success(f"Saved {cache_lnl_file}")
    return lnl_cache
