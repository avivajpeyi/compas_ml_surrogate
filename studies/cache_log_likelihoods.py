from glob import glob

from compas_surrogate.cosmic_integration.universe import Universe
from compas_surrogate.liklelihood.likelihood_cacher import (
    compute_and_cache_lnl,
)

OUTDIR = "out_muz_sigma0"
GLOB_STR = f"{OUTDIR}/*.npz"

CACHE_LNL_FILE = "cache_lnl.npz"


def main(universes_glob=GLOB_STR, cache_lnl_file=CACHE_LNL_FILE):
    universe_paths = glob(universes_glob)
    # load a set of universes and choose a "true" universe
    observed_uni = Universe.from_npz(universe_paths[0])
    mock_population = observed_uni.sample_possible_event_matrix()
    compute_and_cache_lnl(mock_population, universe_paths, cache_lnl_file)


if __name__ == "__main__":
    main()
