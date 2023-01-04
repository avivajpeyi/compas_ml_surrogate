import os
import unittest

import matplotlib.pyplot as plt
import numpy as np

from compas_surrogate.compas_output_parser.compas_output import CompasOutput
from compas_surrogate.cosmic_integration.universe import Universe
from compas_surrogate.plotting.gif_generator import make_gif


class TestUniverse(unittest.TestCase):
    def setUp(self) -> None:
        self.small_testfile = os.path.join(
            os.path.dirname(__file__),
            "test_data/COMPAS_Output/COMPAS_Output.h5",
        )
        self.co = CompasOutput.from_h5(self.small_testfile)

        LARGEFILE_PATH = "/home/compas-data/h5out_32M.h5"
        SMALLFILE_PATH = "../../quasir_compass_blocks/data/COMPAS_Output.h5"

        if os.path.exists(LARGEFILE_PATH):
            self.testfile = LARGEFILE_PATH
        else:
            self.testfile = SMALLFILE_PATH

        self.outdir = "out_test/universe"
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir, exist_ok=True)

    def test_universe(self):
        # uni = Universe.simulate(self.testfile, SF=[0.01, 2.77, 2.90, 4.70])
        # print("run complete")
        # outfn = uni.save(outdir=self.outdir)
        outfn = "out_test/universe/uni_n11735_sf_0.01_2.77_2.9_4.7.npz"
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
        for i, dSF in enumerate(np.linspace(0.5, 5.7, 30)):
            dSF = np.round(dSF, 2)
            SF = [0.01, 2.77, 2.90, dSF]
            uni = Universe.simulate(self.testfile, SF=SF)
            uni = uni.bin_detection_rate()
            fig = uni.plot_detection_rate_matrix(save=False)
            fig.savefig(
                os.path.join(self.outdir, f"detection_rate_matrix_{i:002}.png")
            )
        make_gif(
            os.path.join(self.outdir, "detection_rate_matrix_*.png"),
            os.path.join(self.outdir, "detection_rate_matrix.gif"),
            duration=150,
            loop=True,
        )

    def test_df(self):
        uni = Universe.simulate(self.testfile, SF=[0.01, 2.77, 2.90, 4.70])
        df = uni.df
        df["x"] = df["z"]
        df["y"] = df["mc"]
        df["z"] = df["rate"]

        # 2D-arrays from DataFrame
        x1 = np.linspace(df["x"].min(), df["x"].max(), len(df["x"].unique()))
        y1 = np.linspace(df["y"].min(), df["y"].max(), len(df["y"].unique()))
        x2, y2 = np.meshgrid(x1, y1)

        # Interpolate unstructured D-dimensional data.
        from scipy.interpolate import griddata

        z2 = griddata((df["x"], df["y"]), df["z"], (x2, y2), method="cubic")

        # Ready to plot
        fig = plt.figure()
        plt.contourf(x2, y2, z2)
        fig.savefig(os.path.join(self.outdir, "test_df.png"))
