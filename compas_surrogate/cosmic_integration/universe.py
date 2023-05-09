import io
import os
import time
from contextlib import redirect_stdout
from typing import Dict, Optional, Union

import emcee
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units
from compas_python_utils.cosmic_integration.FastCosmicIntegration import (
    find_detection_rate,
)
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from scipy.interpolate import RectBivariateSpline

from compas_surrogate.cosmic_integration.star_formation_paramters import (
    DEFAULT_SF_PARAMETERS as DEFAULT,
)
from compas_surrogate.logger import logger
from compas_surrogate.plotting import safe_savefig

MC_RANGE = [1, 40]
Z_RANGE = [0, 0.6]
O2_DURATION = 0.5  # in yrs

CMAP = "inferno"


class MockPopulation:
    def __init__(self, rate2d, mcz, universe):
        self.rate2d = rate2d
        self.mcz = mcz
        self.universe = universe
        self.muz: float = universe.muz
        self.sigma0: float = universe.sigma0
        self.aSF: float = universe.SF[0]
        self.bSF: float = universe.SF[1]
        self.cSF: float = universe.SF[2]
        self.dSF: float = universe.SF[3]

    @property
    def n_events(self) -> int:
        return len(self.mcz)

    def plot(self, save=True, outdir=".", fname=""):
        fig = self.universe.plot_detection_rate_matrix(save=False, titles=False)
        axes = fig.get_axes()
        axes[0].scatter(
            self.mcz[:, 1], self.mcz[:, 0], s=15, c="dodgerblue", marker="*", alpha=0.95
        )
        fig.suptitle(f"Mock population ({self.n_events} blue stars)")
        axes[1].set_title(self.universe.param_str, fontsize=7)
        if save or fname:
            if fname == "":
                fname = self.universe._get_fname(outdir, extra="mock_events", ext="png")
            safe_savefig(fig, fname)
        return fig

    def hist_mock_2d_events(self):
        u: Universe = self.universe
        mc_bins, z_bins = u.get_mc_and_z_bins()
        hist2d_mcz = np.histogram2d(
            self.mcz[:, 1], self.mcz[:, 0], bins=[z_bins, mc_bins]
        )[0].T
        hist1d_mc = np.histogram(self.mcz[:, 0], bins=mc_bins)[0]
        hist1d_z = np.histogram(self.mcz[:, 1], bins=z_bins)[0]
        hist1d_z = (hist1d_z / np.sum(hist1d_z)) * np.sum(u.redshift_rate)
        hist1d_mc = (hist1d_mc / np.sum(hist1d_mc)) * np.sum(u.chirp_mass_rate)
        return hist2d_mcz, hist1d_z, hist1d_mc

    def plot_hist(self, save=True, outdir=".", fname=""):
        fig = self.universe.plot_detection_rate_matrix(save=False, titles=False)
        axes = fig.get_axes()
        ax_2d, ax_top, ax_right = axes[0], axes[1], axes[2]
        cbar_ax = axes[3]
        u: Universe = self.universe
        h_2d, h_x, h_y = self.hist_mock_2d_events()

        cbar = ax_2d.pcolormesh(
            u.redshifts,
            u.chirp_masses,
            h_2d,
            cmap="viridis",
            norm="linear",
        )
        fig.colorbar(
            cbar,
            cax=cbar_ax,
            orientation="vertical",
            label="COUNT",
            aspect=5,
        )
        cbar_ax.tick_params(labelsize=8, length=0)
        cbar_ax.yaxis.set_major_locator(LinearLocator(5))
        cbar_ax.yaxis.set_ticks_position("left")

        # plot 1d histograms ensuring normalized to "redshift_rate"

        kwgs = dict(color="dodgerblue", alpha=0.8, ls="dashed")
        ax_top.step(u.redshifts, h_x, **kwgs)
        ax_right.step(h_y, u.chirp_masses, **kwgs)

        fig.suptitle(f"Hist mock-population ({self.n_events} mock events)")
        mse = self.mse_2d_grid()
        axes[1].set_title(f"Grid MSE: {mse:.2e}", fontsize=7)
        if save or fname:
            if fname == "":
                fname = self.universe._get_fname(outdir, extra="mock_hist", ext="png")
            safe_savefig(fig, fname)
        return fig

    def mse_2d_grid(self):
        hist2d, _, _ = self.hist_mock_2d_events()
        rate2d = self.universe.detection_rate

        hist2d = hist2d.ravel() / np.sum(hist2d.ravel())
        rate2d = rate2d.ravel() / np.sum(rate2d.ravel())

        # mean squre error
        mse = np.square(np.subtract(rate2d, hist2d)).mean()
        return mse

    @property
    def param_list(self) -> np.array:
        return self.universe.param_list

    def __repr__(self):
        return f"MockPopulation({self.universe})"

    def __dict__(self):
        return {
            "rate2d": self.rate2d,
            "mcz": self.mcz,
            **self.universe.__dict__(),
        }

    def save(self, fname):
        data = {k: np.asarray(v) for k, v in self.__dict__().items()}
        np.savez(fname, **data)
        return fname

    @classmethod
    def from_npz(cls, fname):
        data = np.load(fname)
        uni = Universe.from_dict(data)
        return cls(universe=uni, mcz=data["mcz"], rate2d=data["rate2d"])


class Universe:
    def __init__(
        self,
        compas_h5_path,
        detection_rate,
        n_systems,
        redshifts,
        chirp_masses,
        SF,
        muz=DEFAULT["muz"],
        sigma0=DEFAULT["sigma0"],
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
    def simulate(
        cls,
        compas_h5_path,
        SF=None,
        muz=DEFAULT["muz"],
        sigma0=DEFAULT["sigma0"],
        outdir=".",
    ):
        """Create a Universe object from a COMPAS h5 file (run cosmic integrator)

        Requires the COMPAS h5 file to have the following datasets:
            BSE_Common_Envelopes
            BSE_Double_Compact_Objects
            BSE_System_Parameters
        """
        if SF is None:
            SF = [
                DEFAULT["aSF"],
                DEFAULT["bSF"],
                DEFAULT["cSF"],
                DEFAULT["dSF"],
            ]
        assert len(SF) == 4, "SF must be a list of length 4"

        logger.debug(
            f"Post-processing {compas_h5_path} with SF={SF}, muz={muz}, sigma0={sigma0}"
        )

        trap = io.StringIO()
        with redirect_stdout(trap):
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
        logger.debug(f"Cosmic integrator logs:\n{trap.getvalue()}")

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

        logger.debug(
            f"{uni.n_systems} simulated systems in {uni.detection_rate.shape} bins, completed in {runtime:.2f} s"
        )
        return uni

    def n_detections(self, duration=1)->float:
        """Calculate the number of detections in a given duration (in years)"""
        marginalised_detection_rate = np.nansum(self.detection_rate)
        return marginalised_detection_rate * duration

    @classmethod
    def from_npz(cls, fname):
        """Create a Universe object from a npz file (dont run cosmic integrator)"""
        data = dict(np.load(fname))
        return cls.from_dict(data)

    @staticmethod
    def _valid_keys():
        return [
            "compas_h5_path",
            "n_systems",
            "detection_rate",
            "redshifts",
            "chirp_masses",
            "SF",
            "muz",
            "sigma0",
        ]

    @classmethod
    def from_dict(cls, data: Dict):
        for key in data:
            if data[key].dtype == "<U48":
                data[key] = data[key].item()
        data = {k: data.get(k, None) for k in cls._valid_keys()}
        uni = cls(**data)
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

    @property
    def param_str(self):
        sf = ",".join([f"{s:.3f}" for s in self.SF])
        return f"SF={sf}, muz={self.muz:.2e}, sigma0={self.sigma0:.2e}"

    @property
    def param_list(self) -> np.array:
        return np.array([*self.SF, self.muz, self.sigma0]).flatten()

    @property
    def param_names(self):
        return ["aSF", "bSF", "cSF", "dSF", "muz", "sigma0"]

    def plot_detection_rate_matrix(
        self,
        save: bool = True,
        outdir: str = ".",
        fname: str = "",
        smoothed_2d_data: bool = False,
        titles: bool = True,
        imshow: bool = False,
        scatter_events: Union[bool, np.ndarray] = None,
    ):
        """Plot the detection rate matrix as a 2D heatmap

        Parameters
        ----------
        save : bool, optional
            Save the figure, by default True
        outdir : str, optional
            Output directory, by default "."
        fname : str, optional
            Output filename, by default "uni_{self.param_str}.png"
        smoothed_2d_data : bool, optional
            Plot the smoothed 2D data, by default False
        titles : bool, optional
            Show the titles, by default True
        imshow : bool, optional
            Use imshow instead of pcolormesh, by default False
        scatter_events : array, optional
            Plot the events on top of the detection rate matrix, by default None
            The array should be a 2d array (n_events, 2) where the first column
            is the chirp mass and the second is the redshift

        """
        os.makedirs(outdir, exist_ok=True)
        title_txt = f"Detection Rate Matrix ({self.n_systems} systems)\n"
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

        if not imshow:
            zz, mcc = np.meshgrid(z, mc)
            cbar = ax_2d.pcolormesh(
                zz,
                mcc,
                rate2d,
                cmap=CMAP,
                norm="linear",
                # vmin=np.quantile(rate2d, 0.5),
                # vmax=np.quantile(rate2d, 0.99),
            )
        else:
            cbar = ax_2d.imshow(
                rate2d,
                cmap=CMAP,
                norm="linear",
                vmin=np.quantile(rate2d, 0.5),
                vmax=np.quantile(rate2d, 0.99),
                aspect="auto",
                interpolation="gaussian",
                origin="lower",
                extent=self.zmc_extents,
            )

        if titles:
            fig.suptitle(title_txt)
            ax_top.set_title(self.param_str, fontsize=7)
        ax_2d.set_xlabel("Redshift")
        ax_2d.set_ylabel("Chirp mass ($M_{\odot}$)")
        ax_2d.set_facecolor("black")
        annote = f"Grid: {rate2d.T.shape}\nN det: {self.n_detections(duration=1)}/yr"
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
        kwgs = dict(color="red", lw=1)
        ax_right.step(
            self.chirp_mass_rate,
            self.chirp_masses,
            **kwgs,
        )
        ax_top.step(
            self.redshifts,
            self.redshift_rate,
            **kwgs,
        )
        ax_right.axis("off")
        ax_top.axis("off")

        if scatter_events is not None:
            ax_2d.scatter(
                scatter_events[:, 1],
                scatter_events[:, 0],
                s=15,
                c="dodgerblue",
                marker="*",
                alpha=0.95,
            )

        ax_right.set_ylim(*MC_RANGE)
        ax_2d.set_ylim(*MC_RANGE)
        ax_top.set_xlim(*Z_RANGE)
        ax_2d.set_xlim(*Z_RANGE)

        # remove space between subplots
        fig.subplots_adjust(hspace=0, wspace=0)

        cbar_ax = fig.add_axes([0.9, 0.1, 0.02, 0.6])
        fig.colorbar(
            cbar,
            cax=cbar_ax,
            orientation="vertical",
            label="Rate (yr$^{-1}$)",
            aspect=5,
        )
        cbar_ax.tick_params(labelsize=8, length=0)
        cbar_ax.yaxis.set_major_locator(LinearLocator(2))
        # foramt tick labels to 2 decimal places
        cbar_ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        cbar_ax.yaxis.set_ticks_position("left")

        # set all cbar_ax spline color to white
        for spine in cbar_ax.spines.values():
            spine.set_edgecolor("white")

        if save or fname:
            if fname == "":
                fname = self._get_fname(outdir, extra="det_matrix", ext="png")
            plt.savefig(fname, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

        return fig

    def _get_fname(self, outdir=".", fname="", extra="", ext="npz"):
        if fname == "":
            fname = f"{self.label}"
            if extra:
                fname += f"_{extra}"
        return os.path.join(outdir, f"{fname}.{ext}")

    def get_mc_and_z_bins(
        self,
        num_mc_bins: Optional[int] = 51,
        num_z_bins: Optional[int] = 101,
    ):
        mc, z = self.chirp_masses, self.redshifts
        mc_bins = np.linspace(MC_RANGE[0], MC_RANGE[1], num_mc_bins)
        z_bins = np.linspace(Z_RANGE[0], Z_RANGE[1], num_z_bins)
        return mc_bins, z_bins

    def bin_detection_rate(
        self, num_mc_bins: Optional[int] = 51, num_z_bins: Optional[int] = 101, frac=1.0
    ):

        binned_data = self.detection_rate.copy()
        mc, z = self.chirp_masses, self.redshifts
        mc_bins, z_bins = self.get_mc_and_z_bins(num_mc_bins, num_z_bins)

        # bin rate data
        binned_data = bin_data2d(binned_data, mc, mc_bins, axis=0)
        binned_data = bin_data2d(binned_data, z, z_bins, axis=1)

        # get bin centers
        mc = 0.5 * (mc_bins[1:] + mc_bins[:-1])
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
            muz=self.muz,
            sigma0=self.sigma0,
        )
        logger.debug(f"Binning data {self.detection_rate.shape} -> {binned_data.shape}")
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

        # drop nans and log the number of rows dropped
        n_nans = df.isna().any(axis=1).sum()
        if n_nans > 0:
            logger.warning(f"Dropping {n_nans}/{len(df)} rows with nan values")
            df = df.dropna()

        # check no nan in dataframe
        assert not df.isna().any().any()

        return df

    @property
    def label(self):
        num = self.n_systems
        sf = "sf_" + "_".join(np.array(self.SF).astype(str))
        muz = f"muz_{self.muz}"
        sigma0 = f"sigma0_{self.sigma0}"
        label = f"uni_n{num}_{sf}_{muz}_{sigma0}"
        if self.binned:
            label = f"binned_{label}"
        return label

    @staticmethod
    def create_label(n_systems, SF, muz, sigma0, binned=False):
        sf = "sf_" + "_".join(np.array(SF).astype(str))
        muz = f"muz_{muz}"
        sigma0 = f"sigma0_{sigma0}"
        label = f"uni_n{n_systems}_{sf}_{muz}_{sigma0}"
        if binned:
            label = f"binned_{label}"
        return label

    @property
    def zmc_extents(self):
        return Z_RANGE + MC_RANGE

    def save(self, outdir=".", fname="") -> str:
        """Save the Universe object to a npz file, return the filename"""
        if fname == "":
            fname = self._get_fname(outdir)

        if outdir == "." and os.path.dirname(fname) != "":
            outdir = os.path.dirname(fname)

        if outdir != ".":
            os.makedirs(outdir, exist_ok=True)

        data = {k: np.asarray(v) for k, v in self.__dict__().items()}

        np.savez(fname, **data)
        return fname

    def __dict__(self):
        """Return a dictionary of the Universe object"""
        return dict(
            compas_h5_path=self.compas_h5_path,
            n_systems=self.n_systems,
            detection_rate=self.detection_rate,
            redshifts=self.redshifts,
            chirp_masses=self.chirp_masses,
            SF=self.SF,
            muz=self.muz,
            sigma0=self.sigma0,
        )

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

    def sample_possible_event_matrix(
        self,
        n_obs: int = None,
    ) -> MockPopulation:
        """Make a fake detection matrix with the same shape as the universe"""
        # FIXME: draw from the detection rate distribution using poisson distributions
        if n_obs is None:
            n_obs = self.n_detections()
        rate2d = np.zeros(self.detection_rate.shape)
        event_mcz = self.sample_observations(n_obs)
        for mc, z in event_mcz:
            mc_bin, z_bin = self.get_matrix_bin_idx(mc, z)
            rate2d[mc_bin, z_bin] += 1

        return MockPopulation(rate2d=rate2d, mcz=event_mcz, universe=self)

    def get_matrix_bin_idx(self, mc, z):
        mc_bin = np.argmin(np.abs(self.chirp_masses - mc))
        z_bin = np.argmin(np.abs(self.redshifts - z))
        return mc_bin, z_bin

    def prob_of_mcz(self, mc, z):
        mc_bin, z_bin = self.get_matrix_bin_idx(mc, z)
        return self.detection_rate[mc_bin, z_bin] / self.n_detections()


def bin_data2d(data2d, data1d, bins, axis=0):
    """Bin data2d and data1d along the given axis using the given 1d bins"""

    num_bins = len(bins)
    assert num_bins != data2d.shape[axis], "More bins than data-rows!"

    bin_data1d_falls_in = np.digitize(data1d, bins)
    assert len(bin_data1d_falls_in) == len(data1d), "Something went wrong with binning"
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
    clean = True
    outdir = "out"
    uni_file = f"{outdir}/uni.npz"
    if not os.path.exists(uni_file) or clean:
        uni = Universe.simulate(PATH, SF=[0.01, 2.77, 2.90, 4.70], muz=-0.3)
        uni_binned = uni.bin_detection_rate()
        uni_binned.save(fname=uni_file)
    uni_binned = Universe.from_npz(uni_file)
    uni_binned.plot_detection_rate_matrix(outdir=outdir)
    mock_pop = uni_binned.sample_possible_event_matrix()
    mock_pop.plot(outdir=outdir)
