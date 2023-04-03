import os
import unittest
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from conftest import LARGE_FILE

from compas_surrogate.cosmic_integration.universe import Universe
from compas_surrogate.data_generation.likelihood_cacher import _get_lnl_and_param_uni

test_datapath = LARGE_FILE
outdir = "bootstrap_testing"
unifn = os.path.join(outdir, "universe.npz")

if os.path.exists(unifn) is False:
    uni = Universe.simulate(test_datapath, SF=[0.01, 2.77, 2.90, 4.70])
    outfn = uni.save(fname=unifn)
uni = Universe.from_npz(unifn)
uni.plot_detection_rate_matrix(outdir=outdir)
binned_uni = uni.bin_detection_rate()

mock_uni = Universe.from_hdf5(
    "/home/avaj040/Documents/projects/compas_ml_surrogate/studies/two_params/det_matrix.h5",
    10,
).sample_possible_event_matrix()


lnl = _get_lnl_and_param_uni(binned_uni, mock_uni.mcz)[0]
print(lnl)
