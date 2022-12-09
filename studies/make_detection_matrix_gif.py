import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from compas_surrogate.cosmic_integration.universe import Universe
from compas_surrogate.plotting.gif_generator import make_gif


def get_compas_output_fname():
    LARGEFILE_PATH = "/home/compas-data/h5out_32M.h5"
    SMALLFILE_PATH = "../../quasir_compass_blocks/data/COMPAS_Output.h5"

    if os.path.exists(LARGEFILE_PATH):
        testfile = LARGEFILE_PATH
    else:
        testfile = SMALLFILE_PATH
    return testfile


def main():
    fname = get_compas_output_fname()
    outdir = "out_universe"
    os.makedirs(outdir, exist_ok=True)

    for i, dSF in enumerate(tqdm(np.linspace(0.5, 6, 50), desc="Universes")):
        dSF = np.round(dSF, 2)
        SF = [0.01, 2.77, 2.90, dSF]
        uni = Universe.simulate(fname, SF=SF)
        fig = uni.plot_detection_rate_matrix(save=False)
        fig.savefig(os.path.join(outdir, f"detection_rate_matrix_{i:002}.png"))
        plt.close(fig)
        uni = uni.bin_detection_rate()
        fig = uni.plot_detection_rate_matrix(save=False)
        fig.savefig(
            os.path.join(outdir, f"binned_detection_rate_matrix_{i:002}.png")
        )
        plt.close(fig)

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


if __name__ == "__main__":
    main()
