"""Module to run COMPAS simulations and generate data for surrogate model"""

import glob
import os

import h5py
import numpy as np
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
    sf_params = dict(aSF=SF[0], dSF=SF[-1], mu_z=muz, sigma_0=sigma0)
    uni = Universe.from_compas_output(
        compas_path=compas_h5_path,
        cosmological_parameters=sf_params,
        max_detectable_redshift=0.6,
        redshift_bins=np.linspace(0, 0.6, 100),
        chirp_mass_bins=np.linspace(3, 40, 50),
        outdir=outdir,
    )
    uni.save(fname=fname)

    if save_images:
        fig = uni.plot()
        fig.savefig(os.path.join(outdir, f"{uni.label}.png"))


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
    logger.info(f"Compiling {n} matrices into hdf file --> {fname}")

    base_uni = Universe.from_npz(npz_files[0])

    with h5py.File(fname, "w") as f:
        f.attrs["compas_path"] = str(base_uni.compas_h5_path)
        f.attrs["n_systems"] = base_uni.n_systems
        f.attrs["n_bbh"] = base_uni.n_bbh
        f.attrs["redshifts"] = base_uni.redshift_bins
        f.attrs["chirp_masses"] = base_uni.chirp_mass_bins
        f.attrs["parameter_labels"] = base_uni.para
        f.create_dataset("detection_matricies", (n, *base_uni.detection_rate.shape))
        f.create_dataset("parameters", (n, *base_uni.param_list.shape))
        for i in tqdm(range(n), "Writing matrices to hdf file"):
            uni = Universe.from_npz(npz_files[i])
            f["detection_matricies"][i, :, :] = uni.detection_rate
            f["parameters"][i, :] = uni.param_list
        f.close()
        filesize_in_gb = os.path.getsize(fname) / 1e9
    logger.success(f"Saved hdf file ({filesize_in_gb} GB)!")


def get_universe_closest_to_parameters(detection_matrix_fn, parameters):
    """
    Get the universe with the closest parameters to the given parameters
    :param detection_matrix_fn: path to detection matrix hdf file
    :param parameters: parameters to match
    :return: Universe object
    """
    with h5py.File(detection_matrix_fn, "r") as f:
        parameters = f["parameters"][:]
        f.close()

    # find the index of the universe with the closest parameters
    idx = np.argmin(np.linalg.norm(parameters - parameters, axis=1))
    uni = Universe.from_hdf5(detection_matrix_fn, idx=idx)
    return uni


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

    if os.path.isfile(save_h5_fname):
        logger.info(
            f"HDF file {save_h5_fname} already exists, skipping matrix generation"
        )
        return

    sf_sample_fname = os.path.join(outdir, "sf_samples.csv")
    if os.path.isfile(sf_sample_fname):
        sf_samples = pd.read_csv(sf_sample_fname)
        n = len(sf_samples)
        logger.info(f"Loading {n} SF samples from {sf_sample_fname}")
        sf_samples = sf_samples[parameters]
        # convert to List[Dict]
        sf_samples = sf_samples.to_dict("records")
    else:
        sf_samples = draw_star_formation_samples(
            n,
            parameters=parameters,
            as_list=True,
            custom_ranges=custom_ranges,
            grid=grid_parameterspace,
        )
        # save sf samples list
        pd.DataFrame(sf_samples).to_csv(
            os.path.join(outdir, "sf_samples.csv"), index=False
        )

    n_proc = get_num_workers()
    logger.info(
        f"Generating matricies (with {n_proc} threads for {n} SF samples with parameters {parameters}"
    )

    fnames = [f"{outdir}/uni_{i}.npz" for i in range(n)]
    args = ([compas_h5_path] * n, sf_samples, [save_images] * n, [outdir] * n, fnames)
    process_map(generate_matrix, *args, max_workers=n_proc, chunksize=n_proc)

    if save_images:
        logger.info("Making GIFs")
        generate_gifs(outdir=outdir)

    if save_h5_fname != "":
        compile_matricies_into_hdf(os.path.join(outdir, "*.npz"), fname=save_h5_fname)
