import os
import unittest

import numpy as np

from compas_surrogate.compas_output_parser.compas_output import CompasOutput
from compas_surrogate.cosmic_integration.universe import Universe


class TestUniverse(unittest.TestCase):
    def setUp(self) -> None:
        self.small_testfile = os.path.join(
            os.path.dirname(__file__),
            "test_data/COMPAS_Output/COMPAS_Output.h5",
        )
        self.testfile = "../../quasir_compass_blocks/data/COMPAS_Output.h5"
        self.co = CompasOutput.from_h5(self.small_testfile)

        self.outdir = "out_test/universe"
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir, exist_ok=True)

    def test_universe(self):
        uni = Universe.from_compas_h5(
            self.testfile, SF=[0.01, 2.77, 2.90, 4.70]
        )
        print("run complete")
        outfn = uni.save(outdir=self.outdir)
        new_uni = Universe.from_npz(outfn)
        # check that new_uni.dco_masses is the same as uni.dco_masses
        self.assertTrue(np.allclose(uni.chirp_masses, new_uni.chirp_masses))
        uni.plot_detection_rate_matrix(
            fname=os.path.join(self.outdir, "detection_rate_matrix_orig.png")
        )
        new_uni.plot_detection_rate_matrix(
            fname=os.path.join(self.outdir, "detection_rate_matrix_new.png")
        )

    def test_different_sf_universes(self):
        for dSF in [4.7 * i for i in range(1, 11)]:
            SF = [0.01, 2.77, 2.90, dSF]
            uni = Universe.from_compas_h5(self.testfile, SF=SF)
            uni.plot_detection_rate_matrix(outdir=self.outdir)
            print("Done with SF = ", SF)
