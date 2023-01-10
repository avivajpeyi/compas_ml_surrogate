import os

import matplotlib.pyplot as plt

from compas_surrogate.cosmic_integration.star_formation_paramters import (
    draw_star_formation_samples,
)

OUTDIR = "out_test"
MAKE_PLOT = True


def test_sampler():
    parameters = ["aSF", "dSF"]
    samples = draw_star_formation_samples(1000, parameters)
    assert samples["aSF"].shape == (1000,)
    assert samples["dSF"].shape == (1000,)
    os.makedirs(OUTDIR, exist_ok=True)
    plt.plot(samples["aSF"], samples["dSF"], ".", color="tab:red", alpha=0.1)
    samples = draw_star_formation_samples(30, parameters)
    plt.plot(samples["aSF"], samples["dSF"], ".", color="tab:blue", zorder=1)
    plt.grid()
    plt.savefig(os.path.join(OUTDIR, "test_sf_samples.png"))
