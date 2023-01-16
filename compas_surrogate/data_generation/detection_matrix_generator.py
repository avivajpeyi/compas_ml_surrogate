"""Module to run COMPAS simulations and generate data for surrogate model"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm.contrib.concurrent import process_map

from compas_surrogate.cosmic_integration.star_formation_paramters import (
    draw_star_formation_samples,
)
from compas_surrogate.cosmic_integration.universe import Universe
from compas_surrogate.logger import logger
from compas_surrogate.plotting.gif_generator import make_gif
from compas_surrogate.utils import get_num_workers


def generate_matrix(compas_h5_path, sf_sample, save_images=False, outdir="."):
    SF = [sf_sample.get("aSF", 0.01), 2.77, 2.90, sf_sample.get("dSF", 4.7)]
    muz = sf_sample.get("muz", -0.23)
    sigma0 = sf_sample.get("sigma0", 0.39)
    sf_params = dict(SF=SF, muz=muz, sigma0=sigma0)
    uni = Universe.simulate(compas_h5_path, **sf_params)
    binned_uni = uni.bin_detection_rate()
    binned_uni.save(outdir=outdir)

    if save_images:
        uni.plot_detection_rate_matrix(outdir=outdir)
        binned_uni.plot_detection_rate_matrix(outdir=outdir)


def generate_gifs(outdir="."):
    make_gif(
        os.path.join(outdir, "uni_*.png"),
        os.path.join(outdir, "det_matrix.gif"),
        duration=100,
        loop=True,
    )
    make_gif(
        os.path.join(outdir, "binned_uni_*.png"),
        os.path.join(outdir, "binned_det_matrix.gif"),
        duration=100,
        loop=True,
    )


def generate_set_of_matricies(
    compas_h5_path, n=50, save_images=True, outdir=".", parameters=None
):
    """
    Generate a set of COMPAS detection rate matricies
    :param compas_h5_path: Path to COMPAS h5 file
    :param n: number of matricies to generate
    :param save_images: save images of the matricies
    :param outdir: dir to save data and images
    :param parameters: parameters to draw from for matrix [aSF, dSF, muz, sigma0]
    :return:
    """
    if parameters == None:
        parameters = ["aSF", "dSF", "muz", "sigma0"]

    if outdir != ".":
        os.makedirs(outdir, exist_ok=True)

    sf_samples = draw_star_formation_samples(
        n, parameters=parameters, as_list=True
    )

    logger.info(
        f"Generating matricies (with {get_num_workers()} for {n} SF samples with parameters {parameters}"
    )

    args = ([compas_h5_path] * n, sf_samples, [save_images] * n, [outdir] * n)
    process_map(
        generate_matrix, *args, max_workers=get_num_workers(), chunksize=100
    )

    if save_images:
        logger.info("Making GIFs")
        generate_gifs(outdir=outdir)
