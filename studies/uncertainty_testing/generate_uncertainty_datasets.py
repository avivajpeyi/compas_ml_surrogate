import os
import shutil

import click
import h5py
import numpy as np
from bilby.core.prior import Normal

from compas_surrogate.bootstrap_tools.sample_compas_output import sample_h5
from compas_surrogate.logger import logger

DCO_KEY = "BSE_Double_Compact_Objects"


def get_num_binaries(compas_h5_filepath):
    with h5py.File(compas_h5_filepath, "r") as compas_h5_file:
        return len(compas_h5_file[DCO_KEY]["SEED"])


def get_sample_size(init_n: int):
    return int(Normal(mu=init_n, sigma=np.sqrt(init_n)).sample())


def generate_datasets(in_compas_h5: str, outdir: str, seed: int):
    np.random.seed(seed)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    init_n = get_num_binaries(in_compas_h5)
    sampled_n = get_sample_size(init_n)
    base_fn = os.path.basename(in_compas_h5)

    out_compas_h5 = os.path.join(
        outdir, base_fn.replace(".h5", f"_sampled_seed{seed}.h5")
    )

    if os.path.exists(out_compas_h5):
        logger.error(f"{out_compas_h5} already exists, skipping.")
        return

    logger.info(f"Sampling {in_compas_h5} to {init_n}->{sampled_n}")
    sample_h5(in_compas_h5, out_compas_h5, n=sampled_n, seed_group=DCO_KEY)
    logger.info(
        f"Saved {out_compas_h5} ({os.stat(out_compas_h5).st_size / 1e9:.2f} GB)"
    )


@click.command()
@click.argument("compas_path", type=click.Path(exists=True))
@click.argument("outdir", type=str)
@click.argument("seed", type=int, default=42)
def cli(compas_path, outdir, seed):
    generate_datasets(compas_path, outdir, seed)


if __name__ == "__main__":
    cli()
