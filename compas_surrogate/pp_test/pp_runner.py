import contextlib
import logging
import os
import random
import traceback
import warnings
from functools import partialmethod
from time import time

import h5py
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from compas_surrogate.inference_runner import run_inference

from ..logger import logger

for l in logging.Logger.manager.loggerDict.keys():
    logging.getLogger(l).setLevel(logging.FATAL)

# disable tqdm (for multiprocessing)
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
        sampler="emcee",
    ):
        """
        Parameters
        ----------
        outdir : str
            The output directory for the results
        det_matricies_fname : str
            The name of the det matrices h5 file
        n_training : int
            The number of training samples to use
        n_injections : int
            The number of injections to use
        """
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.det_matricies_fname = det_matricies_fname
        self.n_training = n_training
        self.n_inj = n_injections
        self.sampler = sampler

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
        logger.info(f"Generated injection file: {self.inj_file}")

    def update_inj_status(self, i, runtime: float):
        df = self.get_injections()
        df.loc[df["universe_id"] == i, "runtime"] = runtime
        df.loc[df["universe_id"] == i, "analysis_complete"] = True
        df.to_csv(self.inj_file, index=False)

    def _run_ith_job(self, i):
        """Run inference for a given universe id"""
        outdir = f"{self.outdir}/out_inj_{i}"
        # with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
        t0 = time()
        run_inference(
            outdir=outdir,
            universe_id=i,
            n=self.n_training,
            det_matrix_h5=self.det_matricies_fname,
            sampler=self.sampler,
        )
        self.update_inj_status(i, time() - t0)

    def run(self):
        """Run inference for all injections"""
        df = self.get_injections()
        logger.info(f"Running inference for {len(df)} injections")

        t0 = time()
        for idx in range(len(df)):
            percent_complete = int(idx / len(df) * 100)
            t1 = time()
            if t1 - t0 > 60 * 5:  # print progress every 5 minutes
                logger.info(f"Progress: {percent_complete}%")
                t0 = t1

            i = df.iloc[idx]["universe_id"]
            analysis_complete = df.iloc[idx]["analysis_complete"]
            if analysis_complete:
                continue
            try:
                self._run_ith_job(i)
            except Exception as e:
                logger.error(f"Failed to run inference for i={i}: {e}")
                logger.error(traceback.format_exc())
