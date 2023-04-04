"""Module to run COMPAS simulations and generate data for surrogate model"""

import glob
import os

import h5py
import pandas as pd
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map

from compas_surrogate.cosmic_integration.star_formation_paramters import (
    DEFAULT_SF_PARAMETERS,
    draw_star_formation_samples,
)
from compas_surrogate.cosmic_integration.universe import Universe
from compas_surrogate.logger import logger
from compas_surrogate.plotting.gif_generator import make_gif
from compas_surrogate.utils import get_num_workers


def generate_matrix(compas_h5_path, sf_sample, save_images=False, outdir=".", fname=""):
    """Generate a detection matrix for a given set of star formation parameters"""
    if os.path.isfile(fname):
        logger.debug(f"Skipping {fname} as it already exists")
        return
    SF = [
        sf_sample.get("aSF", DEFAULT_SF_PARAMETERS["aSF"]),
        DEFAULT_SF_PARAMETERS["bSF"],
        DEFAULT_SF_PARAMETERS["cSF"],
        sf_sample.get("dSF", DEFAULT_SF_PARAMETERS["dSF"]),
    ]
    muz = sf_sample.get("muz", DEFAULT_SF_PARAMETERS["muz"])
    sigma0 = sf_sample.get("sigma0", DEFAULT_SF_PARAMETERS["sigma0"])
    sf_params = dict(SF=SF, muz=muz, sigma0=sigma0)
    uni = Universe.simulate(compas_h5_path, **sf_params)
    binned_uni = uni.bin_detection_rate()
    binned_uni.save(outdir=outdir, fname=fname)

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


def compile_matricies_into_hdf(npz_regex, fname="detection_matricies.h5") -> None:
    """
    Compile a set of COMPAS detection rate matricies into a single hdf file
    :param npz_paths: list of paths to npz files
    :param fname: name of output file
    :return: None
    """
    npz_files = glob.glob(npz_regex)
    n = len(npz_files)
    if n == 0:
        raise ValueError(f"No files found with regex: {npz_regex}")
    logger.info(f"Compiling {n} matricies into hdf file --> {fname}")

    base_uni = Universe.from_npz(npz_files[0])

    with h5py.File(fname, "w") as f:
        f.attrs["compas_h5_path"] = str(base_uni.compas_h5_path)
        f.attrs["n_systems"] = base_uni.n_systems
        f.attrs["redshifts"] = base_uni.redshifts
        f.attrs["chirp_masses"] = base_uni.chirp_masses
        f.attrs["parameter_labels"] = base_uni.param_names
        f.create_dataset("detection_matricies", (n, *base_uni.detection_rate.shape))
        f.create_dataset("parameters", (n, *base_uni.param_list.shape))
        for i in tqdm(range(n), "Writing matricies to hdf file"):
            uni = Universe.from_npz(npz_files[i])
            f["detection_matricies"][i, :, :] = uni.detection_rate
            f["parameters"][i, :] = uni.param_list
        f.close()
        filesize_in_gb = os.path.getsize(fname) / 1e9
    logger.success(f"Saved hdf file ({filesize_in_gb} GB)!")


def generate_set_of_matricies(
    compas_h5_path,
    n=50,
    save_images=True,
    outdir=".",
    parameters=None,
    save_h5_fname="detection_matricies.h5",
    custom_ranges=None,
    grid_parameterspace=False,
):
    """
    Generate a set of COMPAS detection rate matricies
    :param compas_h5_path: Path to COMPAS h5 file
    :param n: number of matricies to generate
    :param save_images: save images of the matricies
    :param outdir: dir to save data and images
    :param parameters: parameters to draw from for matrix [aSF, dSF, muz, sigma0]
    :param save_h5_fname: save matricies to hdf file
    :return:
    """
    if parameters == None:
        parameters = ["aSF", "dSF", "muz", "sigma0"]

    if outdir != ".":
        os.makedirs(outdir, exist_ok=True)

    sf_samples = draw_star_formation_samples(
        n,
        parameters=parameters,
        as_list=True,
        custom_ranges=custom_ranges,
        grid=grid_parameterspace,
    )
    # save sf samples list
    pd.DataFrame(sf_samples).to_csv(os.path.join(outdir, "sf_samples.csv"))
    fnames = [f"{outdir}/uni_{i}.npz" for i in range(n)]

    n_proc = get_num_workers()
    logger.info(
        f"Generating matricies (with {n_proc} threads for {n} SF samples with parameters {parameters}"
    )

    args = ([compas_h5_path] * n, sf_samples, [save_images] * n, [outdir] * n, fnames)
    process_map(generate_matrix, *args, max_workers=n_proc, chunksize=n_proc)

    if save_images:
        logger.info("Making GIFs")
        generate_gifs(outdir=outdir)

    if save_h5_fname != "":
        compile_matricies_into_hdf(os.path.join(outdir, "*.npz"), fname=save_h5_fname)
