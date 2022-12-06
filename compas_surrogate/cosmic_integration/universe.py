"""File to run CosmicIntegrator"""
import logging
import os
import time
from argparse import Namespace
from functools import cached_property
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from astropy import units
from skimage import io

from compas_surrogate.cosmic_integration.CosmicIntegration import (
    find_detection_rate,
)

logger = logging.getLogger()


class Universe:
    def __init__(
        self,
        compas_h5_path,
        max_detectable_redshift=1,
        detection_rate=None,
        formation_rate=None,
        merger_rate=None,
        redshifts=None,
        dco_chirp_masses=None,
        SF=None,
    ):

        self.compas_h5_path = compas_h5_path
        self.max_detectable_redshift = max_detectable_redshift
        self.detection_rate = detection_rate
        self.formation_rate = formation_rate
        self.merger_rate = merger_rate
        self.redshifts = redshifts
        self.dco_chirp_masses = dco_chirp_masses

        if SF is None:
            SF = [0.01, 2.77, 2.90, 4.70]
        self.SF = SF

    @classmethod
    def from_compas_h5(cls, compas_h5_path, SF=None):
        """Create a Universe object from a COMPAS h5 file (run cosmic integrator)"""
        if SF:
            assert len(SF) == 4, "SF must be a list of length 4"
        uni = cls(compas_h5_path, SF=SF)
        uni.run_cosmic_integrator()
        return uni

    @classmethod
    def from_npz(cls, fname):
        """Create a Universe object from a npz file (dont run cosmic integrator)"""
        data = dict(np.load(fname))
        for key in data:
            if data[key].dtype == "<U48":
                data[key] = data[key].item()
        return cls(**data)

    def run_cosmic_integrator(self):
        start_CI = time.time()
        (
            self.detection_rate,
            self.formation_rate,
            self.merger_rate,
            self.redshifts,
            COMPAS,
        ) = find_detection_rate(
            path=self.compas_h5_path,
            dco_type="BBH",
            merger_output_filename=None,
            weight_column=None,
            merges_hubble_time=True,
            pessimistic_CEE=True,
            no_RLOF_after_CEE=True,
            max_redshift=10.0,
            max_redshift_detection=self.max_detectable_redshift,
            redshift_step=0.001,
            z_first_SF=10,
            use_sampled_mass_ranges=True,
            m1_min=5 * units.Msun,
            m1_max=150 * units.Msun,
            m2_min=0.1 * units.Msun,
            fbin=0.7,
            aSF=self.SF[0],
            bSF=self.SF[1],
            cSF=self.SF[2],
            dSF=self.SF[3],
            mu0=0.035,
            muz=-0.23,
            sigma0=0.39,
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
        end_CI = time.time()
        print("Time taken for CI: ", end_CI - start_CI)

        sorted_idx = np.argsort(COMPAS.mChirp)
        self.dco_chirp_masses = COMPAS.mChirp[sorted_idx]
        self.detection_rate = self.detection_rate[sorted_idx, :]
        self.merger_rate = self.merger_rate[sorted_idx, :]
        self.formation_rate = self.formation_rate[sorted_idx, :]

    def plot_detection_rate_matrix(
        self, num_chirp_mass_bins=None, fname="", outdir=".", interactive=False
    ):
        """Plot the detection rate matrix"""
        # get midpoints of redshift bins
        z = self.redshifts[self.redshifts < self.max_detectable_redshift]
        mc = self.dco_chirp_masses
        detections = self.detection_rate

        if (len(mc), len(z)) != detections.shape:
            raise ValueError(
                f"Shape of detection rate matrix ({detections.shape}) "
                f"does not match redshift and chirp mass bins ({len(mc)}, {len(z)})"
            )

        # if num_chirp_mass_bins is not None:
        #     # get midpoints of chirp mass bins
        #     mc_bins = np.linspace(mc.min(), mc.max(), num_chirp_mass_bins + 1)
        #     mc = 0.5 * (mc_bins[1:] + mc_bins[:-1])
        #     # sum over chirp mass bins
        #     detections = np.sum(detections.reshape(len(mc_bins) - 1, -1), axis=0)
        #     detections = detections.reshape(1, -1)
        #
        #
        # num_bins = 40
        # bins = np.linspace(low_mc, high_mc, num_bins)
        # mc_bins = np.digitize(mc[sort_idx], bins)
        # binned_rates = np.zeros((num_bins, len(z)))
        # sorted_dr = detection_rate[sort_idx, :]
        #
        # for bii in range(1, num_bins + 1):
        #     mask = mc_bins[mc_bins == bii]
        #     binned_rates[bii - 1] = np.sum(sorted_dr[mask], axis=0)

        title_txt = f"Detection Rate Matrix ({len(mc)} systems)"
        title_txt += f"\nSF: {', '.join(np.array(self.SF).astype(str))}"

        low_mc, high_mc = np.min(mc), np.max(mc)
        low_z, high_z = np.min(z), np.max(z)
        # norm = mpl.colors.Normalize(vmin=np.exp(-100), vmax=np.exp(-12))

        fig = plt.figure(figsize=(5, 5))
        plt.imshow(
            self.detection_rate,
            cmap=plt.cm.hot,
            norm="linear",
            vmin=1e-40,
            vmax=1e-5,
            aspect="auto",
            interpolation="gaussian",
            origin="lower",
            extent=[low_z, high_z, low_mc, high_mc],
        )
        if interactive:
            axis = plt.gca()
            axis.xaxis.set_visible(False)
            axis.yaxis.set_visible(False)
            for spine in ["top", "right", "left", "bottom"]:
                axis.spines[spine].set_visible(False)
        else:
            plt.suptitle(title_txt)
            plt.xlabel("Redshift")
            plt.ylabel("Chirp mass")
            plt.colorbar(label="Detection rate")
        plt.tight_layout()

        if interactive:
            outdir = os.path.join(outdir, ".temp")
            os.makedirs(outdir, exist_ok=True)

        if not outdir == ".":
            fname = os.path.join(outdir, f"{self.label}_det_matrix.png")

        if not fname == "":
            if not interactive:
                plt.savefig(fname)
            else:
                plt.savefig(fname, bbox_inches="tight", pad_inches=0)

        if interactive:
            img_data = io.imread(fname)
            fig = px.imshow(
                img_data,
                labels=dict(
                    x="Redshift", y="Chirp mass", color="Detection rate"
                ),
                title=title_txt,
                # aspect="auto",
                color_continuous_scale="hot",
            )
            # fig.update_xaxes(range=[low_z, high_z])
            # fig.update_yaxes(range=[low_mc, high_mc])
            fig.update_layout(coloraxis_colorbar=dict(title="Detection rate"))
            fig.update_xaxes(showticklabels=False).update_yaxes(
                showticklabels=False
            )
        return fig

    def plot_merger_rate(self):
        """Plot the merger rate"""
        plt.figure()
        plt.plot(self.redshifts, self.merger_rate)
        plt.xlabel("Redshift")
        plt.ylabel("Merger rate")

    def plot_binned_merger_rate(self, bin_width=0.1):
        """Plot the binned merger rate"""
        plt.figure()
        z = self.redshifts
        m = self.merger_rate
        z_bins = np.arange(0, 10, bin_width)
        m_bins = np.zeros_like(z_bins)
        for i, z_bin in enumerate(z_bins):
            m_bins[i] = np.sum(m[(z >= z_bin) & (z < z_bin + bin_width)])
        plt.plot(z_bins, m_bins)
        plt.xlabel("Redshift")
        plt.ylabel("Merger rate")
        plt.yscale("log")

    def bin_data(self, data, num_bins: int):
        mc, z = self.dco_chirp_masses, self.redshifts
        bins = np.linspace(mc.min(), mc.max(), num_bins)
        bin_mc_falls_in = np.digitize(mc, bins)
        assert len(bin_mc_falls_in) == len(mc)
        assert data.shape[0] == len(mc)
        binned_data = np.zeros((num_bins - 1, data.shape[1]))
        for bii in range(1, num_bins):
            mask = bin_mc_falls_in == bii
            logger.debug(
                f"BIN {bii}[{bins[bii - 1]:.2f}, {bins[bii]:.2f}]: "
                f"count {len(mc[mask])} {mc[mask].min(), mc[mask].max()}"
            )
            binned_data[bii - 1] = np.sum(data[mask, :], axis=0)

        binned_mc = 0.5 * (bins[1:] + bins[:-1])

        assert len(binned_mc) == binned_data.shape[0]

        return binned_data, binned_mc, bins

    def save(self, outdir=".", fname="") -> str:
        """Save the Universe object to a npz file, return the filename"""
        data = {k: np.asarray(v) for k, v in self.__dict__().items()}
        if fname == "":
            fname = os.path.join(outdir, f"{self.label}.npz")
        np.savez(fname, **data)
        return fname

    @property
    def label(self):
        num = len(self.dco_chirp_masses)
        sf = "_".join(np.array(self.SF).astype(str))
        return f"uni_dco_n{num}_sf_{sf}"

    def __dict__(self):
        """Return a dictionary of the Universe object"""
        return dict(
            compas_h5_path=self.compas_h5_path,
            max_detectable_redshift=self.max_detectable_redshift,
            detection_rate=self.detection_rate,
            formation_rate=self.formation_rate,
            merger_rate=self.merger_rate,
            redshifts=self.redshifts,
            dco_chirp_masses=self.dco_chirp_masses,
            SF=self.SF,
        )

    def log_likelihood(self, data):
        """Calculate the likelihood of the data given the Universe object"""

        u = np.sum(self.detection_rate)

        if u <= 0.0:
            return -np.inf

        nObs = np.sum(
            data
        )  # do this outside the function and pass it in if you're doing this many times in a loop

        t1 = nObs * np.log(u) - u

        det_shape = self.detection_rate.shape

        p = detections / u
        p += (
            1.0 / float(det_shape.shape[0]) / float(det_shape.shape[1])
        )  # to avoid p = 0.0
        pSum = np.sum(p)
        p /= pSum

        t2 = np.sum(data * np.log(p))

        return t1 + t2
