import os.path

import pandas as pd
import pytest
from bilby.core.prior import PriorDict, TruncatedGaussian, Uniform

from compas_surrogate.plotting.likelihood_plotter import plot_1d_lnl, plot_hacky_1d_lnl

N = 5000

TRUE = 5
TRUE_DICT = dict(x=TRUE, y=TRUE)


@pytest.fixture
def likelihood_data():
    data_range = 0, 10
    samples = PriorDict(dict(x=Uniform(*data_range), y=Uniform(*data_range))).sample(N)
    kwgs = dict(minimum=0, maximum=10, sigma=1, mu=TRUE)
    prior = PriorDict(dict(x=TruncatedGaussian(**kwgs), y=TruncatedGaussian(**kwgs)))
    df = pd.DataFrame(samples)
    return df, prior.ln_prob(samples, axis=0)


def test_plot_1d_lnl(likelihood_data, tmp_path):
    samples, lnl = likelihood_data
    fig = plot_hacky_1d_lnl(samples, lnl, TRUE_DICT)
    fname = f"{tmp_path}/test_plot_1d_lnl.png"
    fig.savefig(fname)
    assert os.path.isfile(fname)
