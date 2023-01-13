import os
from functools import cache

from compas_surrogate.data_generation import generate_set_of_matricies


@cache
def get_compas_output_fname():
    LARGEFILE_PATH = "/home/compas-data/h5out_5M.h5"
    SMALLFILE_PATH = "../../quasir_compass_blocks/data/COMPAS_Output.h5"

    if os.path.exists(LARGEFILE_PATH):
        testfile = LARGEFILE_PATH
    else:
        testfile = SMALLFILE_PATH
    return testfile


OUTDIR = "out_universe_muz"


def main():
    generate_set_of_matricies(
        n=5,
        save_images=True,
        outdir=OUTDIR,
        compas_h5_path=get_compas_output_fname(),
        parameters=["muz"],
    )


if __name__ == "__main__":
    main()
