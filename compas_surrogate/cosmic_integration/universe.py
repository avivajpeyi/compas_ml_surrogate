"""File to run CosmicIntegrator"""
import time
from argparse import Namespace
from functools import cached_property

import numpy as np
from astropy import units

from compas_surrogate.cosmic_integration.CosmicIntegration import (
    find_detection_rate,
)


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
    ):
        self.compas_h5_path = compas_h5_path
        self.max_detectable_redshift = max_detectable_redshift
        self.detection_rate = detection_rate
        self.formation_rate = formation_rate
        self.merger_rate = merger_rate
        self.redshifts = redshifts
        self.dco_chirp_masses = dco_chirp_masses

    @classmethod
    def from_compas_h5(cls, compas_h5_path):
        """Create a Universe object from a COMPAS h5 file (run cosmic integrator)"""
        uni = cls(compas_h5_path)
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
            aSF=0.01,
            bSF=2.77,
            cSF=2.90,
            dSF=4.70,
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
        self.dco_chirp_masses = COMPAS.mChirp

    def plot_detection_rate_matrix(self):
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

    def save(self, fname):
        """Save the Universe object to a npz file"""
        data = {k: np.asarray(v) for k, v in self.__dict__().items()}
        np.savez(fname, **data)

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
        )
