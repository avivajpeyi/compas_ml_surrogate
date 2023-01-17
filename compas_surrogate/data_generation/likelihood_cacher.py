from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from compas_surrogate.cosmic_integration.universe import (
    MockPopulation,
    Universe,
)
from compas_surrogate.liklelihood import ln_likelihood
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
    h5_file: h5py.File, idx: int, observed_mcz: np.ndarray
):
    uni = Universe.from_hdf5(h5_file, idx)
    return _get_lnl_and_param_uni(uni, observed_mcz)


def compute_and_cache_lnl(
    mock_population: MockPopulation,
    cache_lnl_file: str,
    hf5_path: Optional[str] = "",
    universe_paths: Optional[List] = None,
):
    if universe_paths is not None:
        n = len(universe_paths)
        logger.info(f"Starting LnL computation for {n} universes")
        lnl_and_param_list = process_map(
            _get_lnl_and_param_from_npz,
            universe_paths,
            [mock_population.mcz] * n,
            desc="Computing likelihoods",
            max_workers=get_num_workers(),
            chunksize=100,
        )
    elif hf5_path is not None:
        f = h5py.File(hf5_path, "r")
        n = len(f["parameters"])
        logger.info(f"Starting LnL computation for {n} universes")
        lnl_and_param_list = [
            _get_lnl_and_param_from_h5(f, i, mock_population.mcz)
            for i in tqdm(
                range(n),
                desc="Computing likelihoods",
            )
        ]
    else:
        raise ValueError("Must provide either hf5_path or universe_paths")

    lnl_and_param_list = np.array(lnl_and_param_list)
    np.savez(
        cache_lnl_file,
        lnl=lnl_and_param_list[:, 0],
        params=lnl_and_param_list[:, 1:],
        true_params=mock_population.param_list,
    )
    logger.success(f"Saved {cache_lnl_file}")


def load_lnl_cache(npz_path: str) -> Dict[str, np.ndarray]:
    data = np.load(npz_path)
    data_dict = dict(
        aSF=data["params"][:, 0],
        bSF=data["params"][:, 1],
        cSF=data["params"][:, 2],
        dSF=data["params"][:, 3],
        muz=data["params"][:, 4],
        sigma0=data["params"][:, 5],
        lnl=data["lnl"],
    )
    df = pd.DataFrame(data_dict)
    init_len = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    logger.info(f"Loaded {npz_path} --> {len(df)}/{init_len} non-nan rows")
    data_dict = df.to_dict("list")

    param_names = ["aSF", "bSF", "cSF", "dSF", "muz", "sigma0"]
    data_dict["true_params"] = {
        p: data["true_params"][i] for i, p in enumerate(param_names)
    }
    return data_dict
