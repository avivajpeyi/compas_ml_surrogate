from compas_surrogate.pp_test import PPrunner, PPresults

import random
import datetime

# get datetime stamp short version
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

OUTDIR = f"out_pp_{now}"
H5 = "det_matrix.h5"
INJ_FILE = f"{OUTDIR}/injection_ids.csv"
N_TRAINING = 500
random.seed(1)


def main():
    runner = PPrunner(OUTDIR, H5, N_TRAINING)
    runner.generate_injection_file(n=N_TRAINING)
    runner.run()
    pp_results = PPresults.from_results(f"{OUTDIR}/out*/*.json")
    pp_results.plot()


if __name__ == "__main__":
    main()
