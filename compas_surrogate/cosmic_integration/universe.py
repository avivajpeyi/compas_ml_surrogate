import logging
import os
import time
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units
from matplotlib.gridspec import GridSpec
from scipy.interpolate import RectBivariateSpline

from compas_surrogate.cosmic_integration.CosmicIntegration import (
    find_detection_rate,
)

logger = logging.getLogger()

MC_RANGE = [1, 40]
Z_RANGE = [0, 0.6]
O2_DURATION = 0.5  # in yrs


class Universe:
    def __init__(
        self,
        compas_h5_path,
        detection_rate,
        n_systems,
        redshifts,
        chirp_masses,
        SF,
        muz=-0.23,
        sigma0=0.39,
        ci_runtime=np.inf,
        binned=False,
    ):

        self.compas_h5_path = compas_h5_path
        self.detection_rate = detection_rate
        self.chirp_mass_rate = np.sum(detection_rate, axis=1)
        self.redshift_rate = np.sum(detection_rate, axis=0)
        self.n_systems = n_systems
        self.redshifts = redshifts
        self.chirp_masses = chirp_masses
        self.binned = binned
        self.ci_runtime = ci_runtime
        self.SF = SF
        self.muz = muz
        self.sigma0 = sigma0

    @staticmethod
    def does_savefile_exist(self, outdir="."):
        fname = self._get_fname(outdir)
        return os.path.exists(fname)

    @classmethod
    def simulate(cls, compas_h5_path, SF=None, muz=-0.23, sigma0=0.39):
        """Create a Universe object from a COMPAS h5 file (run cosmic integrator)"""
        if SF is None:
            SF = [0.01, 2.77, 2.90, 4.70]
        assert len(SF) == 4, "SF must be a list of length 4"

        start_time = time.time()
        (
            detection_rate,
            formation_rate,
            merger_rate,
            redshifts,
            COMPAS,
        ) = find_detection_rate(
            path=compas_h5_path,
            dco_type="BBH",
            merger_output_filename=None,
            weight_column=None,
            merges_hubble_time=True,
            pessimistic_CEE=True,
            no_RLOF_after_CEE=True,
            max_redshift=10.0,
            max_redshift_detection=max(Z_RANGE),
            redshift_step=0.001,
            z_first_SF=10,
            use_sampled_mass_ranges=True,
            m1_min=5 * units.Msun,
            m1_max=150 * units.Msun,
            m2_min=0.1 * units.Msun,
            fbin=0.7,
            aSF=SF[0],
            bSF=SF[1],
            cSF=SF[2],
            dSF=SF[3],
            mu0=0.035,
            muz=muz,
            sigma0=sigma0,
            sigmaz=0.0,
            alpha=0.0,
            min_logZ=-12.0,
            max_logZ=0.0,
            step_logZ=0.01,
            sensitivity="O1",
            snr_threshold=8,
            Mc_max=300.0,
            Mc_step=0.1,
            eta_max=0.25,
            eta_step=0.01,
            snr_max=1000.0,
            snr_step=0.1,
        )
        runtime = time.time() - start_time

        sorted_idx = np.argsort(COMPAS.mChirp)
        chirp_masses = COMPAS.mChirp[sorted_idx]
        redshift_mask = redshifts < max(Z_RANGE)
        redshifts = redshifts[redshift_mask]
        if (len(chirp_masses), len(redshifts)) != detection_rate.shape:
            raise ValueError(
                f"Shape of detection rate matrix ({detection_rate.shape}) "
                f"does not match chirp mass + redshift bins ({(len(chirp_masses), len(redshifts))})"
            )

        uni = cls(
            compas_h5_path,
            chirp_masses=chirp_masses,
            n_systems=len(chirp_masses),
            detection_rate=detection_rate[sorted_idx, :],
            redshifts=redshifts,
            ci_runtime=runtime,
            SF=SF,
            muz=muz,
            sigma0=sigma0,
        )

        print(f"Time taken for CI: {runtime:.2f} s")
        print(
            f"{uni.n_systems} simulated systems in {uni.detection_rate.shape} bins"
        )
        return uni

    def n_detections(self, duration=1):
        """Calculate the number of detections in a given duration (in years)"""
        return int(np.sum(self.detection_rate) * duration)

    @classmethod
    def from_npz(cls, fname):
        """Create a Universe object from a npz file (dont run cosmic integrator)"""
        data = dict(np.load(fname))
        for key in data:
            if data[key].dtype == "<U48":
                data[key] = data[key].item()
        valid_keys = [
            "compas_h5_path",
            "n_systems",
            "detection_rate",
            "redshifts",
            "chirp_masses",
            "SF",
        ]
        data = {k: data[k] for k in valid_keys}
        return cls(**data)

    def plot_detection_rate_matrix(
        self, save=True, outdir=".", smoothed_2d_data=False
    ):

        title_txt = f"Detection Rate Matrix ({self.n_systems} systems)"
        title_txt += f"\nSF: {', '.join(np.array(self.SF).astype(str))}"
        if self.binned:
            title_txt = "Binned " + title_txt
        if smoothed_2d_data:
            title_txt = "Smoothed " + title_txt

        z, mc, rate2d = self.redshifts, self.chirp_masses, self.detection_rate
        low_mc, high_mc = np.min(mc), np.max(mc)
        low_z, high_z = np.min(z), np.max(z)

        if smoothed_2d_data:
            z = np.linspace(low_z, high_z, 20)
            mc = np.linspace(low_mc, high_mc, 20)
            rate2d = self.get_detection_rate_spline()(mc, z)

        fig = plt.figure(figsize=(5, 5))
        gs = GridSpec(4, 4)

        ax_2d = fig.add_subplot(gs[1:4, 0:3])
        ax_top = fig.add_subplot(gs[0, 0:3])
        ax_right = fig.add_subplot(gs[1:4, 3])

        ax_2d.imshow(
            rate2d,
            cmap=plt.cm.hot,
            norm="linear",
            vmin=np.quantile(rate2d, 0.5),
            vmax=np.quantile(rate2d, 0.99),
            aspect="auto",
            interpolation="gaussian",
            origin="lower",
            extent=self.zmc_extents,
        )

        fig.suptitle(title_txt)
        ax_2d.set_xlabel("Redshift")
        ax_2d.set_ylabel("Chirp mass")
        ax_2d.set_facecolor("black")
        annote = (
            f"Grid: {rate2d.shape}\nN det: {self.n_detections(duration=1)}/yr"
        )
        ax_2d.annotate(
            annote,
            xy=(1, 0),
            xycoords="axes fraction",
            xytext=(-5, 5),
            textcoords="offset points",
            ha="right",
            va="bottom",
            color="white",
        )
        ax_right.plot(self.chirp_mass_rate, self.chirp_masses, color="red")
        ax_top.plot(self.redshifts, self.redshift_rate, color="red")
        ax_right.axis("off")
        ax_top.axis("off")

        ax_right.set_ylim(*MC_RANGE)
        ax_2d.set_ylim(*MC_RANGE)
        ax_top.set_xlim(*Z_RANGE)
        ax_2d.set_xlim(*Z_RANGE)

        plt.tight_layout()

        if save:
            fname = self._get_fname(outdir, extra="det_matrix", ext="png")
            plt.savefig(fname, bbox_inches="tight", pad_inches=0)

        return fig

    def _get_fname(self, outdir=".", fname="", extra="", ext="npz"):
        if fname == "":
            fname = f"{self.label}"
            if extra:
                fname += f"_{extra}"
        return os.path.join(outdir, f"{fname}.{ext}")

    def bin_detection_rate(
        self, num_mc_bins: Optional[int] = 51, num_z_bins: Optional[int] = 101
    ):

        binned_data = self.detection_rate.copy()
        mc, z = self.chirp_masses, self.redshifts

        if num_mc_bins is not None:
            mc_bins = np.linspace(MC_RANGE[0], MC_RANGE[1], num_mc_bins)
            binned_data = bin_data2d(binned_data, mc, mc_bins, axis=0)
            mc = 0.5 * (mc_bins[1:] + mc_bins[:-1])
        if num_z_bins is not None:
            z_bins = np.linspace(Z_RANGE[0], Z_RANGE[1], num_z_bins)
            binned_data = bin_data2d(binned_data, z, z_bins, axis=1)
            z = 0.5 * (z_bins[1:] + z_bins[:-1])

        assert (len(mc), len(z)) == binned_data.shape

        new_uni = Universe(
            compas_h5_path=self.compas_h5_path,
            detection_rate=binned_data,
            n_systems=self.n_systems,
            redshifts=z,
            chirp_masses=mc,
            binned=True,
            ci_runtime=self.ci_runtime,
            SF=self.SF,
        )
        print(
            f"Binning data {self.detection_rate.shape} -> {binned_data.shape}"
        )
        return new_uni

    def get_detection_rate_spline(self):
        z, mc, rate2d = self.redshifts, self.chirp_masses, self.detection_rate
        min_rate = np.quantile(rate2d, 0.6)
        max_rate = np.quantile(rate2d, 0.99)
        rate2d = np.clip(rate2d, min_rate, max_rate)
        return RectBivariateSpline(mc, z, rate2d)

    def get_detection_rate_dataframe(self):
        z, mc = self.redshifts, self.chirp_masses
        rate = self.detection_rate.ravel()
        zz, mcc = np.meshgrid(z, mc)
        df = pd.DataFrame({"z": zz.ravel(), "mc": mcc.ravel(), "rate": rate})

        # check no nan in dataframe
        assert not df.isna().any().any()

        return df

    def save(self, outdir=".") -> str:
        """Save the Universe object to a npz file, return the filename"""
        data = {k: np.asarray(v) for k, v in self.__dict__().items()}
        fname = self._get_fname(outdir)
        np.savez(fname, **data)
        return fname

    @property
    def label(self):
        num = self.n_systems
        sf = "_".join(np.array(self.SF).astype(str))

        label = f"uni_n{num}_sf_{sf}"
        if self.binned:
            label = f"binned_{label}"
        return label

    @property
    def zmc_extents(self):
        return Z_RANGE + MC_RANGE

    def __dict__(self):
        """Return a dictionary of the Universe object"""
        return dict(
            compas_h5_path=self.compas_h5_path,
            n_systems=self.n_systems,
            detection_rate=self.detection_rate,
            redshifts=self.redshifts,
            chirp_masses=self.chirp_masses,
            SF=self.SF,
        )

    def sample_observations(self, n_obs: int = None):
        if n_obs is None:
            n_obs = self.n_detections()
        df = self.get_detection_rate_dataframe()
        df = df.sort_values("rate", ascending=False)
        n_events = df.sample(weights=df.rate, n=n_obs)
        return n_events[["mc", "z"]].values

    def sample_possible_event_matrix(self, n_obs: int = None):
        """Make a fake detection matrix with the same shape as the universe"""
        # FIXME: draw from the detection rate distribution using poisson distributions
        if n_obs is None:
            n_obs = self.n_detections()
        rate2d = np.zeros(self.detection_rate.shape)
        event_mcz = self.sample_observations(n_obs)
        for mc, z in event_mcz:
            mc_bin, z_bin = self.get_matrix_bin_idx(mc, z)
            rate2d[mc_bin, z_bin] += 1
        return rate2d, event_mcz

    def get_matrix_bin_idx(self, mc, z):
        mc_bin = np.argmin(np.abs(self.chirp_masses - mc))
        z_bin = np.argmin(np.abs(self.redshifts - z))
        return mc_bin, z_bin

    def prob_of_mcz(self, mc, z):
        mc_bin, z_bin = self.get_matrix_bin_idx(mc, z)
        return self.detection_rate[mc_bin, z_bin] / self.n_detections()


def plot_event_matrix_on_universe_detection_matrix(
    universe: Universe, true_rate2d: np.ndarray = None
):
    """Plot the fake detection matrix"""
    if true_rate2d is None:
        true_rate2d._ = universe.sample_possible_event_matrix()

    universe.plot_detection_rate_matrix()
    fig = plt.gcf()
    axes = fig.get_axes()
    axes[0].imshow(
        true_rate2d,
        origin="lower",
        extent=universe.zmc_extents,
        aspect="auto",
        cmap="tab10",
        alpha=1.0 * (true_rate2d > 0),
    )
    return fig


def bin_data2d(data2d, data1d, bins, axis=0):
    """Bin data2d and data1d along the given axis using the given 1d bins"""

    num_bins = len(bins)
    assert num_bins != data2d.shape[axis], "More bins than data-rows!"

    bin_data1d_falls_in = np.digitize(data1d, bins)
    assert len(bin_data1d_falls_in) == len(
        data1d
    ), "Something went wrong with binning"
    assert data2d.shape[axis] == len(data1d), "Data2d and bins do not match"

    # bin data
    binned_data_shape = list(data2d.shape)
    binned_data_shape[axis] = num_bins - 1
    binned_data = np.zeros(binned_data_shape)
    for bii in range(1, num_bins):
        mask = bin_data1d_falls_in == bii
        masked_data = data2d.take(indices=np.where(mask)[0], axis=axis)
        if axis == 0:
            binned_data[bii - 1, :] = np.sum(masked_data, axis=axis)
        else:
            binned_data[:, bii - 1] = np.sum(masked_data, axis=axis)
    return binned_data


if __name__ == "__main__":
    PATH = "/Users/avaj0001/Documents/projects/compas_dev/quasir_compass_blocks/data/COMPAS_Output.h5"
    from tqdm.auto import tqdm

    uni_file = "uni.npz"
    if not os.path.exists(uni_file):
        uni = Universe.simulate(PATH, SF=[0.01, 2.77, 2.90, 4.70])
        uni.save(outdir="out")
    uni = Universe.from_npz(uni_file)
    uni_binned = uni.bin_detection_rate()
    fig = uni_binned.plot_detection_rate_matrix(outdir="out")
