"""Module to run COMPAS simulations and generate data for surrogate model"""

import glob
import os

import h5py
from tqdm.auto import tqdm
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
        generate_matrix, *args, max_workers=get_num_workers(), chunksize=10
    )

    if save_images:
        logger.info("Making GIFs")
        generate_gifs(outdir=outdir)


def compile_matricies_into_hdf(
    npz_regex, fname="detection_matricies.h5"
) -> None:
    """
    Compile a set of COMPAS detection rate matricies into a single hdf file
    :param npz_paths: list of paths to npz files
    :param fname: name of output file
    :return: None
    """
    npz_files = glob.glob(npz_regex)
    n = len(npz_files)
    logger.info(f"Compiling {n} matricies into hdf file --> {fname}")

    base_uni = Universe.from_npz(npz_files[0])

    with h5py.File(fname, "w") as f:
        f.attrs["compas_h5_path"] = str(base_uni.compas_h5_path)
        f.attrs["n_systems"] = base_uni.n_systems
        f.attrs["redshifts"] = base_uni.redshifts
        f.attrs["chirp_masses"] = base_uni.chirp_masses
        f.attrs["parameter_labels"] = base_uni.param_names
        f.create_dataset(
            "detection_matricies", (n, *base_uni.detection_rate.shape)
        )
        f.create_dataset("parameters", (n, *base_uni.param_list.shape))
        for i in tqdm(range(n), "Writing matricies to hdf file"):
            uni = Universe.from_npz(npz_files[i])
            f["detection_matricies"][i, :, :] = uni.detection_rate
            f["parameters"][i, :] = uni.param_list
        f.close()
