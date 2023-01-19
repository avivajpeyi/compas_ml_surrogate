import os
import random
from glob import glob

import h5py
import matplotlib.pyplot as plt

from compas_surrogate.cosmic_integration.universe import Universe
from compas_surrogate.data_generation.likelihood_cacher import (
    LikelihoodCache,
    compute_and_cache_lnl,
)
from compas_surrogate.logger import logger
from compas_surrogate.surrogate.models import DeepGPModel

OUTDIR = "out_surr"
CACHE_LNL_FILE = f"{OUTDIR}/cache_lnl.npz"
H5 = "det_matrix.h5"


def generate_cache(cache_lnl_file=CACHE_LNL_FILE) -> LikelihoodCache:
    if os.path.exists(cache_lnl_file):
        logger.info(f"Loading cache from {cache_lnl_file}")
        return LikelihoodCache.from_npz(cache_lnl_file)
    else:
        logger.info(f"Generating cache {cache_lnl_file}")
        h5_file = h5py.File(H5, "r")
        observed_uni = Universe.from_hdf5(h5_file, random.randint(0, 1000))
        mock_population = observed_uni.sample_possible_event_matrix()
        mock_population.plot(save=True, outdir=OUTDIR)
        lnl_cache = compute_and_cache_lnl(
            mock_population, cache_lnl_file, h5_path=H5
        )
        lnl_cache.plot(fname=f"{OUTDIR}/lnl_cache_{mock_population}.png")
    return lnl_cache


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    cache = generate_cache()
    model = DeepGPModel()
    model.train(cache.params, cache.lnl.reshape(-1, 1))
    model.save(OUTDIR)
    logger.success("Done")


if __name__ == "__main__":
    main()
