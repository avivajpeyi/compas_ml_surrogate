import io
import os
import shutil
import time
from contextlib import redirect_stdout
from typing import Dict, Optional, Union

import emcee
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units
from compas_python_utils.cosmic_integration.binned_cosmic_integrator.detection_matrix import (
    DetectionMatrix,
)
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from scipy.interpolate import RectBivariateSpline

from compas_surrogate.cosmic_integration.star_formation_paramters import (
    DEFAULT_SF_PARAMETERS as DEFAULT,
)
from compas_surrogate.logger import logger
from compas_surrogate.plotting import safe_savefig

MC_RANGE = [3, 40]
Z_RANGE = [0, 0.6]
O2_DURATION = 0.5  # in yrs

CMAP = "inferno"


class Universe(DetectionMatrix):
    def n_detections(self, duration=1) -> float:
        """Calculate the number of detections in a given duration (in years)"""
        marginalised_detection_rate = np.nansum(self.rate_matrix)
        return marginalised_detection_rate * duration

    @classmethod
    def from_npz(cls, fname):
        """Create a Universe object from a npz file (dont run cosmic integrator)"""
        data = dict(np.load(fname, allow_pickle=True))
        return cls.from_dict(data)

    @property
    def param_list(self) -> np.array:
        return np.array(self.cosmological_parameters.values()).flatten()

    @property
    def param_names(self):
        return list(self.cosmological_parameters.keys())

    @classmethod
    def from_dict(cls, data: Dict):
        obj_types = [
            "compas_path",
            "cosmological_parameters",
            "n_systems",
            "n_bbh",
        ]
        for key in data:
            if key in obj_types:
                data[key] = data[key].item()
        uni = cls(
            compas_path=data["compas_path"],
            cosmological_parameters=data["cosmological_parameters"],
            rate_matrix=data["rate_matrix"],
            chirp_mass_bins=data["chirp_mass_bins"],
            redshift_bins=data["redshift_bins"],
            n_systems=data["n_systems"],
            n_bbh=data["n_bbh"],
            outdir=data.get("outdir", "."),
            bootstrapped_rate_matrices=data.get("bootstrapped_rate_matrices", None),
        )
        logger.debug(f"Loaded cached uni with: {uni.param_str}")
        return uni

    @classmethod
    def from_hdf5(
        cls,
        h5file: Union[h5py.File, str],
        idx: int = None,
        search_param: Dict[str, float] = {},
    ):
        """Create a Universe object from a hdf5 file (dont run cosmic integrator)"""
        data = {}
        common_keys = [
            "compas_h5_path",
            "n_systems",
            "redshifts",
            "chirp_masses",
        ]

        h5file_opened = False
        if isinstance(h5file, str):
            h5file = h5py.File(h5file, "r")
            h5file_opened = True

        for key in common_keys:
            data[key] = h5file.attrs.get(key, None)
            if data[key] is None:
                logger.warning(
                    f"Could not find {key} in hdf5 file. Attributes avail: {h5file.attrs.keys()}"
                )

        if idx is None:
            params = h5file["parameters"]
            search_val = [
                search_param["aSF"],
                search_param["bSF"],
                search_param["cSF"],
                search_param["dSF"],
                search_param["muz"],
                search_param["sigma0"],
            ]
            # get index of the closest match
            idx = np.argmin(np.sum((params[:] - search_val) ** 2, axis=1))
            logger.debug(f"Found closest match at index {idx}")

        data["detection_rate"] = h5file["detection_matricies"][idx]
        params = h5file["parameters"][idx]
        data["SF"] = params[:4]
        data["muz"] = params[4]
        data["sigma0"] = params[5]
        uni = cls(**data)
        logger.debug(f"Loaded cached uni with: {uni.param_str}")

        if h5file_opened:
            h5file.close()

        return uni

    def _get_fname(self, outdir=".", fname="", extra="", ext="npz"):
        if fname == "":
            fname = f"{self.label}"
            if extra:
                fname += f"_{extra}"
        return os.path.join(outdir, f"{fname}.{ext}")

    def __dict__(self) -> Dict:
        return self.to_dict()

    def get_detection_rate_dataframe(self):
        z, mc = self.redshift_bins, self.chirp_mass_bins
        rate = self.rate_matrix.ravel()
        zz, mcc = np.meshgrid(z, mc)
        df = pd.DataFrame({"z": zz.ravel(), "mc": mcc.ravel(), "rate": rate})

        # drop nans and log the number of rows dropped
        n_nans = df.isna().any(axis=1).sum()
        if n_nans > 0:
            logger.warning(f"Dropping {n_nans}/{len(df)} rows with nan values")
            df = df.dropna()

        # check no nan in dataframe
        assert not df.isna().any().any()

        return df

    def save(self, fname="") -> str:
        """Save the Universe object to a npz file, return the filename"""
        super().save()
        if fname != "":
            orig_fname = f"{self.outdir}/{self.label}.h5"
            shutil.move(orig_fname, fname)

    def __repr__(self):
        return f"<Universe: [{self.n_systems} systems], {self.param_str}>"

    def sample_observations(
        self,
        n_obs: int = None,
        sample_using_detection_rate_as_weights: bool = True,
        sample_using_emcee: bool = False,
    ) -> np.ndarray:
        if n_obs is None:
            n_obs = self.n_detections()
        if sample_using_detection_rate_as_weights:
            df = self.get_detection_rate_dataframe()
            df = df.sort_values("rate", ascending=False)
            if np.sum(df.rate) > 0:
                n_events = df.sample(
                    weights=df.rate, n=int(n_obs), random_state=0, replace=True
                )
            else:
                n_events = df.sample(n=int(n_obs), random_state=0)
            return n_events[["mc", "z"]].values
        elif sample_using_emcee:
            raise NotImplementedError

    def get_matrix_bin_idx(self, mc, z):
        mc_bin = np.argmin(np.abs(self.chirp_mass_bins - mc))
        z_bin = np.argmin(np.abs(self.redshift_bins - z))
        return mc_bin, z_bin

    def prob_of_mcz(self, mc, z):
        mc_bin, z_bin = self.get_matrix_bin_idx(mc, z)
        return self.rate_matrix[mc_bin, z_bin] / self.n_detections()

    def get_bootstrapped_uni(self, i: int):
        """Creates a new Uni using the ith bootstrapped rate matrix"""
        assert (
            i < self.n_bootstraps
        ), f"i={i} is larger than the number of bootstraps {self.n_bootstraps}"
        return Universe(
            compas_path=self.compas_path,
            cosmological_parameters=self.cosmological_parameters,
            rate_matrix=self.bootstrapped_rate_matrices[i],
            chirp_mass_bins=self.chirp_mass_bins,
            redshift_bins=self.redshift_bins,
            n_systems=self.n_systems,
            n_bbh=self.n_bbh,
            outdir=self.outdir,
        )

    @property
    def n_bootstraps(self):
        return len(self.bootstrapped_rate_matrices)


class MockPopulation:
    def __init__(self, rate2d, mcz, universe: Universe):
        self.rate2d = rate2d
        self.mcz = mcz
        self.universe = universe

    @classmethod
    def sample_possible_event_matrix(
        cls,
        universe: Universe,
        n_obs: int = None,
    ) -> "MockPopulation":
        """Make a fake detection matrix with the same shape as the universe"""
        # FIXME: draw from the detection rate distribution using poisson distributions
        if n_obs is None:
            n_obs = universe.n_detections()
        rate2d = np.zeros(universe.rate_matrix.shape)
        event_mcz = universe.sample_observations(n_obs)
        for mc, z in event_mcz:
            mc_bin, z_bin = universe.get_matrix_bin_idx(mc, z)
            rate2d[mc_bin, z_bin] += 1

        return MockPopulation(rate2d=rate2d, mcz=event_mcz, universe=universe)

    @property
    def n_events(self) -> int:
        return len(self.mcz)

    def plot(self):
        fig = self.universe.plot()
        axes = fig.get_axes()
        axes[0].scatter(
            self.mcz[:, 1], self.mcz[:, 0], s=15, c="dodgerblue", marker="*", alpha=0.95
        )
        fig.suptitle(f"Mock population ({self.n_events} blue stars)")
        axes[1].set_title(self.universe.param_str, fontsize=7)
        return fig

    def __repr__(self):
        return f"MockPopulation({self.universe})"

    def __dict__(self):
        uni_data = self.universe.__dict__()
        uni_data["bootstrapped_rate_matrices"] = None
        return {
            "rate2d": self.rate2d,
            "mcz": self.mcz,
            **uni_data,
        }

    def save(self, fname=""):
        if fname == "":
            fname = f"{self.universe.outdir}/mock_pop.npz"
        np.savez(fname, **self.__dict__())

    @classmethod
    def from_npz(cls, fname):
        data = dict(np.load(fname, allow_pickle=True))
        u = Universe.from_dict(data)
        return cls(rate2d=data["rate2d"], mcz=data["mcz"], universe=u)


if __name__ == "__main__":
    PATH = "/Users/avaj0001/Documents/projects/compas_dev/quasir_compass_blocks/data/COMPAS_Output.h5"
    clean = True
    outdir = "out"
    for i in range(10):
        np.random.seed(i)
        uni_file = f"{outdir}/v{i}.h5"
        uni = Universe.from_compas_output(
            PATH,
            n_bootstrapped_matrices=10,
            outdir=outdir,
            redshift_bins=np.linspace(0, 0.6, 100),
            chirp_mass_bins=np.linspace(3, 40, 50),
        )
        uni.save(fname=uni_file)
        uni = Universe.from_h5(uni_file)

    uni = Universe.from_h5(f"{outdir}/v0.h5")
    mock_pop = MockPopulation.sample_possible_event_matrix(uni)
    mock_pop.save(f"{outdir}/mock_pop.npz")
    mock_pop = MockPopulation.from_npz(f"{outdir}/mock_pop.npz")
    fig = mock_pop.plot()
    fig.savefig(f"{outdir}/mock_pop.png")

    # uni_binned = Universe.from_npz(uni_file)
    # uni_binned.plot_detection_rate_matrix(outdir=outdir)
    # mock_pop = uni_binned.sample_possible_event_matrix()
    # mock_pop.plot(outdir=outdir)
