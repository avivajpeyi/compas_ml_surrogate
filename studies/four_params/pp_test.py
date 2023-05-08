import datetime
import random

from compas_surrogate.pp_test import PPresults, PPrunner

H5 = "det_matrix.h5"
random.seed(1)


def run_analyses_and_make_pp_plot(outdir, h5, n_training, n_injections):
    runner = PPrunner(
        outdir=outdir,
        det_matricies_fname=h5,
        n_training=n_training,
        n_injections=n_injections,
    )
    runner.generate_injection_file(n=n_injections)
    runner.run()
    pp_results = PPresults.from_results(f"{outdir}/out*/*.json")
    pp_results.plot(f"{outdir}/pp_plot.png")


def now():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


if __name__ == "__main__":
    for n_train in [500, 1000, 2000, 2500, 3000]:
        run_analyses_and_make_pp_plot(
            outdir=f"out_pp_ntrain_{n_train}_{now()}",
            h5=H5,
            n_training=n_train,
            n_injections=500,
        )
