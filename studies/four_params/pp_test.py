import contextlib
import logging
import os
import random
import warnings
from functools import partialmethod
from time import time

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from compas_surrogate.inference_runner import run_inference

for l in logging.Logger.manager.loggerDict.keys():
    logging.getLogger(l).setLevel(logging.FATAL)


# disable tqdm
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


OUTDIR = "out_pp"
H5 = "det_matrix.h5"
INJ_FILE = f"{OUTDIR}/injection_ids.csv"
N_TRAINING = 500
random.seed(1)


def generate_injection_file(n=100):
    h5 = h5py.File(H5, "r")
    total = len(h5["detection_matricies"])
    # draw 100 random universes:
    universe_ids = np.random.randint(0, total, n)
    analysis_complete = [False for _ in range(n)]
    df = pd.DataFrame(
        {"universe_id": universe_ids, "analysis_complete": analysis_complete}
    )
    df.to_csv(INJ_FILE, index=False)


def update_injection_file_with_complete(i):
    df = pd.read_csv(INJ_FILE)
    df.loc[df["universe_id"] == i, "analysis_complete"] = True
    df.to_csv(INJ_FILE, index=False)


def main():
    if not os.path.exists(INJ_FILE):
        os.makedirs(OUTDIR, exist_ok=True)
        generate_injection_file(n=100)
    inj_df = pd.read_csv(INJ_FILE)
    const_kwgs = dict(n=N_TRAINING, det_matrix_h5=H5)
    print("Running inference...\n\n")
    avg_times = []
    for idx in range(len(inj_df)):
        avgtime = np.mean(avg_times)
        prnt_txt = f"{idx:02d}/{len(inj_df)} [{avgtime:.2f}s/it]"
        print(prnt_txt)

        i = inj_df.iloc[idx]["universe_id"]
        analysis_complete = inj_df.iloc[idx]["analysis_complete"]
        if analysis_complete:
            continue
        outdir = f"{OUTDIR}/out_inj_{i}"
        try:
            start = time()
            # disable all logging
            with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
                run_inference(outdir=outdir, universe_id=i, **const_kwgs)
            end = time()
            avg_times.append(end - start)
            update_injection_file_with_complete(i)

        except Exception as e:
            print(f"Failed to run inference for i={i}: {e}")


if __name__ == "__main__":
    main()
