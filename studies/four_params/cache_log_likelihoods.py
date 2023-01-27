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


def generate_cache(outdir, n_samp=100) -> LikelihoodCache:
    cache_file = f"{outdir}/cache_lnl.npz"
    if os.path.exists(cache_file):
        logger.info(f"Loading cache from {cache_file}")
        lnl_cache = LikelihoodCache.from_npz(cache_file)
    else:
        logger.info(f"Generating cache {cache_file}")
        h5_file = h5py.File(H5, "r")
        observed_uni = Universe.from_hdf5(h5_file, random.randint(0, 1000))
        logger.info(f"Uni idx {observed_uni}")
        mock_population = observed_uni.sample_possible_event_matrix()
        mock_population.plot(save=True, outdir=OUTDIR)
        lnl_cache = compute_and_cache_lnl(
            mock_population, cache_file, h5_path=H5
        )
        lnl_cache.plot(fname=cache_file.replace(".npz", ".png"))
    lnl_cache = lnl_cache.sample(n_samp)
    return lnl_cache


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    cache = generate_cache(outdir=OUTDIR)
    model = DeepGPModel()
    model.train(cache.params, cache.lnl.reshape(-1, 1), verbose=True)
    model.save(OUTDIR)
    logger.success("Done")


if __name__ == "__main__":
    main()
