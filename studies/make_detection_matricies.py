import os
from functools import cache
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from compas_surrogate.cosmic_integration.universe import Universe
from compas_surrogate.plotting.gif_generator import make_gif

OUTDIR = "out_universe"


@cache
def get_compas_output_fname():
    LARGEFILE_PATH = "/home/compas-data/h5out_32M.h5"
    SMALLFILE_PATH = "../../quasir_compass_blocks/data/COMPAS_Output.h5"

    if os.path.exists(LARGEFILE_PATH):
        testfile = LARGEFILE_PATH
    else:
        testfile = SMALLFILE_PATH
    return testfile


def generate_matrix(dSF, save_images=False, outdir=OUTDIR):
    print(f"... Starting  {dSF} ...")
    SF = [0.01, 2.77, 2.90, dSF]
    uni = Universe.simulate(get_compas_output_fname(), SF=SF)

    binned_uni = uni.bin_detection_rate()
    binned_uni.save(outdir=outdir)

    if save_images:
        fig = uni.plot_detection_rate_matrix(save=False)
        fig.savefig(os.path.join(outdir, f"detection_rate_matrix_{dSF}.png"))
        plt.close(fig)
        fig = binned_uni.plot_detection_rate_matrix(save=False)
        fig.savefig(
            os.path.join(outdir, f"binned_detection_rate_matrix_{dSF}.png")
        )
        plt.close(fig)
    print(f"... Finished computing for {dSF} --> {uni.label} ...")


def generate_set_of_matricies(n=50, save_images=True):
    outdir = "out_universe"
    os.makedirs(outdir, exist_ok=True)

    dSF_list = np.round(np.linspace(0.5, 6, n), 5)
    print(f"Generating matricies for SF: {dSF_list}")
    num_workers = cpu_count()
    if num_workers > 64:
        num_workers = 64
    elif num_workers < 16:
        num_workers = 4
    print(f"Generating matricies (with {num_workers} workers)")
    with Pool(num_workers) as pool:
        for dSF in tqdm(dSF_list):
            pool.apply_async(generate_matrix, args=(dSF,))
        pool.close()
        pool.join()

    if save_images:
        make_gif(
            os.path.join(outdir, "detection_rate_matrix_*.png"),
            os.path.join(outdir, "detection_rate_matrix.gif"),
            duration=100,
            loop=True,
        )
        make_gif(
            os.path.join(outdir, "binned_detection_rate_matrix_*.png"),
            os.path.join(outdir, "binned_detection_rate_matrix.gif"),
            duration=100,
            loop=True,
        )


def main():
    generate_set_of_matricies(n=500, save_images=False)


if __name__ == "__main__":
    main()
