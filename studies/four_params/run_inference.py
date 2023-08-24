import random

from compas_surrogate.inference_runner import run_inference

OUTDIR = "out_surr"
H5 = "det_matrix.h5"
random.seed(1)


def main():
    n_pts = [250]
    for n in n_pts:
        outdir = f"{OUTDIR}/out_n_{n}"
        run_inference(
            outdir=outdir,
            n=n,
            cache_outdir=OUTDIR,
            det_matrix_h5=H5,
            universe_id=5000,
            clean=False,
            sampler="emcee",
        )


if __name__ == "__main__":
    main()
