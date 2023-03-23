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

from ..logger import logger

for l in logging.Logger.manager.loggerDict.keys():
    logging.getLogger(l).setLevel(logging.FATAL)

# disable tqdm
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

OUTDIR = "out_pp"
H5 = "det_matrix.h5"
INJ_FILE = f"{OUTDIR}/injection_ids.csv"
N_TRAINING = 500
random.seed(1)


class PPrunner:
    def __init__(
        self,
        outdir: str,
        det_matricies_fname: str,
        n_training: int = 500,
        n_injections: int = 100,
    ):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.det_matricies_fname = det_matricies_fname
        self.n_training = n_training
        self.n_inj = n_injections

    def get_injections(self) -> pd.DataFrame:
        """Get the injection dataframe from the inj file"""
        if not self.inj_file_exists:
            self.generate_injection_file(n=self.n_inj)
        return pd.read_csv(self.inj_file)

    @property
    def inj_file(self) -> str:
        return f"{self.outdir}/injection_ids.csv"

    @property
    def inj_file_exists(self) -> bool:
        return os.path.isfile(self.inj_file)

    def generate_injection_file(self, n=100):
        """Generate a csv file with n random universe ids"""
        logger.info(f"Generating injection file with n={n}")
        assert self.inj_file_exists is False, "inj file already exists"
        h5 = h5py.File(self.det_matricies_fname, "r")
        total = len(h5["detection_matricies"])
        assert n < total, "n must be less than total number of universes"
        df = pd.DataFrame(
            dict(
                universe_id=np.random.randint(0, total, n),
                analysis_complete=[False for _ in range(n)],
                runtime=[np.nan for _ in range(n)],
            )
        )
        df.to_csv(self.inj_file, index=False)

    def update_inj_status(self, i, runtime: float):
        df = self.get_injections()
        df.loc[df["universe_id"] == i, "runtime"] = runtime
        df.loc[df["universe_id"] == i, "analysis_complete"] = True
        df.to_csv(self.inj_file, index=False)

    def _run_ith_job(self, i):
        """Run inference for a given universe id"""
        outdir = f"{self.outdir}/out_inj_{i}"
        with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
            t0 = time()
            run_inference(
                outdir=outdir,
                universe_id=i,
                n=self.n_training,
                det_matrix_h5=self.det_matricies_fname,
            )
        self.update_inj_status(i, time() - t0)

    def run(self):
        """Run inference for all injections"""
        df = self.get_injections()
        logger.info(f"Running inference for {len(df)} injections")
        for idx in tqdm(range(len(df))):
            i = df.iloc[idx]["universe_id"]
            analysis_complete = df.iloc[idx]["analysis_complete"]
            if analysis_complete:
                continue
            try:
                self._run_ith_job(i)
            except Exception as e:
                logger.error(f"Failed to run inference for i={i}: {e}")
