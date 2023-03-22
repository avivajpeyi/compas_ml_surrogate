from argparse import ArgumentParser

from compas_surrogate.data_generation.detection_matrix_generator import (
    compile_matricies_into_hdf,
    generate_set_of_matricies,
)
from compas_surrogate.logger import logger


def cli_matrix_generation():
    """CLI for generating COMPAS detection rate matricies"""
    parser = ArgumentParser(description="Generate COMPAS detection rate matricies")
    parser.add_argument("--compas_h5_path", type=str, help="path to COMPAS h5 file")
    parser.add_argument(
        "--n", type=int, default=100, help="number of matricies to generate"
    )
    parser.add_argument(
        "--save_images",
        default=False,
        action="store_true",
        help="save images of matricies",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="out_matricies",
        help="output directory for matricies",
    )
    parser.add_argument(
        "--parameters",
        type=str,
        nargs="+",
        default=["aSF", "dSF", "muz", "sigma0"],
        help="parameters to draw from for matrix [aSF, dSF, muz, sigma0]",
    )
    parser.add_argument(
        "--save_h5_fname",
        type=str,
        default="detection_matricies.h5",
        help="save matricies to hdf file",
    )
    args = parser.parse_args()
    logger.info(f"Running matrix generation with args: {args}")
    generate_set_of_matricies(
        compas_h5_path=args.compas_h5_path,
        n=args.n,
        save_images=args.save_images,
        outdir=args.outdir,
        parameters=args.parameters,
        save_h5_fname=args.save_h5_fname,
    )


def cli_compile_h5():
    """CLI for compiling COMPAS detection rate matricies into a single hdf file"""
    parser = ArgumentParser(
        description="Compile COMPAS detection rate matricies into a single hdf file"
    )
    parser.add_argument("--npz_regex", type=str, help="regex to match npz files")
    parser.add_argument("--fname", type=str, help="name of output h5 file")
    args = parser.parse_args()
    logger.info(f"Running h5 complier args: {args}")
    compile_matricies_into_hdf(
        npz_regex=args.npz_regex,
        fname=args.fname,
    )
