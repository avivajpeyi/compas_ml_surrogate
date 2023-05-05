import datetime
import random

from compas_surrogate.pp_test import PPresults, PPrunner

# get datetime stamp short version
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

OUTDIR = f"out_pp_{now}"
H5 = "det_matrix.h5"
INJ_FILE = f"{OUTDIR}/injection_ids.csv"
N_TRAINING = 5000
N_INJ = 500
random.seed(1)


def main():
    runner = PPrunner(OUTDIR, H5, n_training=N_TRAINING, n_injections=N_INJ)
    runner.generate_injection_file(n=N_INJ)
    runner.run()
    pp_results = PPresults.from_results(f"{OUTDIR}/out*/*.json")
    pp_results.plot()


if __name__ == "__main__":
    main()
