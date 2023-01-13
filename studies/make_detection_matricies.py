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


def main():
    kwgs = dict(
        n=15, save_images=True, compas_h5_path=get_compas_output_fname()
    )
    generate_set_of_matricies(
        outdir="out_asf",
        parameters=["aSF"],
        **kwgs,
    )
    generate_set_of_matricies(
        outdir="out_dsf",
        parameters=["dSF"],
        **kwgs,
    )
    generate_set_of_matricies(
        outdir="out_muz",
        parameters=["muz"],
        **kwgs,
    )
    generate_set_of_matricies(
        outdir="out_sig",
        parameters=["sigma0"],
        **kwgs,
    )


if __name__ == "__main__":
    main()
