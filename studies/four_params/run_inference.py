import random
import warnings

import numpy as np

from compas_surrogate.inference_runner import run_inference

OUTDIR = "out_surr"
H5 = "det_matrix.h5"
random.seed(1)


def main():
    n_pts = np.linspace(100, 1000, 10).astype(int)
    n_pts = [100, 250, 500, 1000]
    for n in n_pts:
        outdir = f"{OUTDIR}/out_n_{n}"
        run_inference(
            outdir=outdir,
            n=n,
            cache_outdir=OUTDIR,
            det_matrix_h5=H5,
            universe_id=5000,
            clean=False,
        )


if __name__ == "__main__":
    main()