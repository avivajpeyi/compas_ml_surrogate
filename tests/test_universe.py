import os
import unittest
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

from compas_surrogate.cosmic_integration.universe import Universe
from compas_surrogate.plotting.gif_generator import make_gif

PLOT = False


def test_universe(test_datapath, tmp_path):
    # fails as testfile is too small --> not enough DCOs
    uni = Universe.simulate(test_datapath, SF=[0.01, 2.77, 2.90, 4.70])
    outfn = uni.save(outdir=tmp_path)
    new_uni = Universe.from_npz(outfn)
    assert np.allclose(uni.chirp_masses, new_uni.chirp_masses)
    mock_uni = uni.sample_possible_event_matrix()
    if PLOT:
        uni.plot_detection_rate_matrix(outdir=tmp_path)
        mock_uni.plot(outdir=tmp_path)
        make_gif(
            os.path.join(tmp_path, "*.png"),
            os.path.join(tmp_path, "nonsense.gif"),
            duration=2,
            loop=True,
        )
    assert False
