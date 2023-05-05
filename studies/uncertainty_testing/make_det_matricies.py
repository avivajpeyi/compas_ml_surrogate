import glob
import os

import click
import numpy as np

from compas_surrogate.data_generation.detection_matrix_generator import (
    generate_set_of_matricies,
)
from compas_surrogate.logger import logger


@click.command()
@click.option("--compas_file_regex", type=str, required=True)
@click.option("--outdir", type=str, required=True)
@click.option("-n", type=int, default=10)
def make_matricies_for_different_results(compas_file_regex, outdir, n=10):
    np.random.seed(0)

    compas_files = glob.glob(compas_file_regex)
    if len(compas_files) == 0:
        raise ValueError(f"No files found matching {compas_file_regex}")
    logger.info(f"Found {len(compas_files)} files matching {compas_file_regex}")

    os.makedirs(outdir, exist_ok=True)

    for i, f in enumerate(compas_files):
        logger.info(f"Generating matrix for {f}")
        label = os.path.basename(f).split(".h5")[0].split("_")[-1]

        generate_set_of_matricies(
            compas_h5_path=f,
            n=n,
            save_images=False,
            outdir=os.path.join(outdir, f"out_{label}"),
            parameters=["muz", "sigma0"],
            save_h5_fname=f"{outdir}/matricies_{label}.h5",
            custom_ranges={"muz": (-0.48, -0.45), "sigma0": (0.17, 0.24)},
            grid_parameterspace=True,
        )


if __name__ == "__main__":
    make_matricies_for_different_results()
