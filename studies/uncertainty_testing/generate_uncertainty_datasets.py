import os
import shutil

import h5py
import numpy as np
from compas_python_utils.h5sample import sample_h5
from tqdm.auto import tqdm

from compas_surrogate.logger import logger

DCO_KEY = "BSE_Double_Compact_Objects"


def get_num_binaries(compas_h5_filepath):
    with h5py.File(compas_h5_filepath, "r") as compas_h5_file:
        return len(compas_h5_file[DCO_KEY]["SEED"])


def generate_datasets(in_compas_h5: str, outdir: str, n_copies: int = 10):
    print("starting")
    np.random.seed(0)
    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=False)
    init_n = get_num_binaries(in_compas_h5)
    sampled_n = int(init_n - np.sqrt(init_n))
    base_fn = os.path.basename(in_compas_h5)
    fn_fmt = os.path.join(outdir, base_fn.replace(".h5", "_downsampled_{}.h5"))
    logger.info(f"Sampling {in_compas_h5} to {init_n}->{sampled_n} ({n_copies} copies)")
    for i in tqdm(range(n_copies), desc="Sampling"):
        out_compas_h5 = fn_fmt.format(i)
        sample_h5(
            in_compas_h5, out_compas_h5, n=sampled_n, replace=False, seed_group=DCO_KEY
        )
        logger.info(f"{out_compas_h5} ({os.stat(out_compas_h5).st_size/1e9:.2f} GB)")


generate_datasets(
    "/fred/oz101/alej_runs/Z_all/COMPAS_Output.h5", "sampled_datasets", 10
)
