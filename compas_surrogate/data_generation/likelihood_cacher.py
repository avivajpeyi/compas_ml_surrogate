import os
import random
from itertools import repeat
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from compas_surrogate.cosmic_integration.universe import MockPopulation, Universe
from compas_surrogate.liklelihood import LikelihoodCache, ln_likelihood
from compas_surrogate.logger import logger
from compas_surrogate.plotting.image_utils import horizontal_concat
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


def _get_lnl_and_param_from_h5(h5_path: h5py.File, idx: int, observed_mcz: np.ndarray):
    uni = Universe.from_hdf5(h5py.File(h5_path, "r"), idx)
    return _get_lnl_and_param_uni(uni, observed_mcz)


def compute_and_cache_lnl(
    mock_population: MockPopulation,
    cache_lnl_file: str,
    h5_path: Optional[str] = "",
    universe_paths: Optional[List] = None,
):
    """
    Compute likelihoods given a Mock Population and universes (either stored in a h5 or the paths to the universe files).
    """
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

    try:
        lnl_and_param_list = np.array(
            process_map(
                *args,
                desc="Computing likelihoods",
                max_workers=get_num_workers(),
                chunksize=100,
                total=n,
            )
        )
    except Exception as e:
        lnl_and_param_list = np.array(
            [
                _get_lnl_and_param_from_h5(h5_path, i, mock_population.mcz)
                for i in tqdm(range(n))
            ]
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
    mock_population.save(f"{os.path.dirname(cache_lnl_file)}/mock_uni.npz")
    logger.success(f"Saved {cache_lnl_file}")
    return lnl_cache


def get_training_lnl_cache(
    outdir,
    n_samp=None,
    det_matrix_h5=None,
    universe_id=None,
    mock_uni=None,
    clean=False,
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

        if mock_uni is None:
            if universe_id is None:
                universe_id = random.randint(0, total_n_det_matricies)
            assert (
                universe_id < total_n_det_matricies
            ), f"Universe id {universe_id} is larger than the number of det matricies {total_n_det_matricies}"
            mock_uni = Universe.from_hdf5(h5_file, universe_id)
        else:
            assert isinstance(mock_uni, Universe)

        mock_population = mock_uni.sample_possible_event_matrix()
        mock_population.plot(save=True, fname=f"{outdir}/injection.png")
        logger.info(
            f"Generating cache {cache_file} using {det_matrix_h5} and universe {universe_id}:{mock_population}"
        )
        lnl_cache = compute_and_cache_lnl(
            mock_population, cache_file, h5_path=det_matrix_h5
        )

    plt_fname = cache_file.replace(".npz", ".png")
    lnl_cache.plot(fname=plt_fname, show_datapoints=True)
    train_plt_fname = plt_fname.replace(".png", "_training.png")
    if n_samp is not None:
        lnl_cache = lnl_cache.sample(n_samp)
    lnl_cache.plot(fname=train_plt_fname, show_datapoints=True)
    horizontal_concat(
        [plt_fname, train_plt_fname], f"{outdir}/cache_pts.png", rm_orig=False
    )

    return lnl_cache
