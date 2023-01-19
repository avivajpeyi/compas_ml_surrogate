import os
import unittest
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

from compas_surrogate.compas_output_parser.compas_output import CompasOutput
from compas_surrogate.cosmic_integration.universe import Universe
from compas_surrogate.plotting.gif_generator import make_gif


class TestUniverse(unittest.TestCase):
    def setUp(self) -> None:
        self.testfile = os.path.join(
            os.path.dirname(__file__),
            "test_data/COMPAS_Output/COMPAS_Output.h5",
        )

        self.outdir = "out_test/universe"
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir, exist_ok=True)

    def test_universe(self):
        # fails as testfile is too small --> not enough DCOs
        uni = Universe.simulate(self.testfile, SF=[0.01, 2.77, 2.90, 4.70])
        outfn = uni.save(outdir=self.outdir)
        new_uni = Universe.from_npz(outfn)
        self.assertTrue(np.allclose(uni.chirp_masses, new_uni.chirp_masses))
        uni.plot_detection_rate_matrix(outdir=self.outdir)
        mock_uni = uni.sample_possible_event_matrix()
        mock_uni.plot(outdir=self.outdir)
        make_gif(
            os.path.join(self.outdir, "*.png"),
            os.path.join(self.outdir, "nonsense.gif"),
            duration=2,
            loop=True,
        )
