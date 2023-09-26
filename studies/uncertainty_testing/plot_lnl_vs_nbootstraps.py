import os

import matplotlib.pyplot as plt
import numpy as np

from compas_surrogate.cosmic_integration.universe import MockPopulation, Universe
from compas_surrogate.data_generation.likelihood_cacher import ln_likelihood

LOCAL_PATH = "/Users/avaj0001/Documents/projects/compas_dev/quasir_compass_blocks/data/COMPAS_Output.h5"
OZ_PATH = "/fred/oz101/avajpeyi/COMPAS_DATA/h5out_32M.h5"

if os.path.exists(OZ_PATH):
    PATH = OZ_PATH
else:
    PATH = LOCAL_PATH

#
outdir = "out1"
N_matrices  = 50
N_bootstraps = 20
for i in range(N_matrices):
    np.random.seed(i)
    uni_file = f"{outdir}/v{i}.h5"

    if not os.path.exists(uni_file):
        uni = Universe.from_compas_output(
            PATH,
            n_bootstrapped_matrices=N_bootstraps,
            outdir=outdir,
            redshift_bins=np.linspace(0, 0.6, 100),
            chirp_mass_bins=np.linspace(3, 40, 50),
            cosmological_parameters=dict(
                aSF=0.01, dSF=4.70, mu_z=-0.3, sigma_z=0, sigma_0=0.4
            ),
        )
        uni.save(fname=uni_file)
        if i == 0:
            fig = uni.plot()
            fig.savefig(f"{outdir}/v0.png")

uni = Universe.from_h5(f"{outdir}/v0.h5")
mock_pop = MockPopulation.sample_possible_event_matrix(uni)
mock_pop.save(f"{outdir}/mock_pop.npz")
fig = mock_pop.plot()
fig.savefig(f"{outdir}/mock_pop.png")

lnl = 0
bootstrapped_lnls = []
for i in range(N_matrices):
    uni_file = f"{outdir}/v{i}.h5"
    uni = Universe.from_h5(uni_file)
    lnl = ln_likelihood(
        mcz_obs=mock_pop.mcz,
        model_prob_func=uni.prob_of_mcz,
        n_model=uni.n_detections(),
        detailed=False,
    )
    print(lnl)
    for j in range(uni.n_bootstraps):
        bootstrap_uni = uni.get_bootstrapped_uni(j)
        bootstrap_lnl = ln_likelihood(
            mcz_obs=mock_pop.mcz,
            model_prob_func=bootstrap_uni.prob_of_mcz,
            n_model=bootstrap_uni.n_detections(),
            detailed=False,
        )
        bootstrapped_lnls.append(bootstrap_lnl)

plt.close("all")
plt.hist(bootstrapped_lnls, color="C0")
plt.axvline(lnl, color="k")
plt.xlabel("lnl")
plt.savefig(f"{outdir}/lnl_hist.png")

# plot of std(lnl) vs number of bootstraps
plt.close("all")
qtiles = []
n_bootstraps = len(bootstrapped_lnls)
for i in range(3, n_bootstraps):
    qtiles.append(np.quantile(bootstrapped_lnls[:i], [0.16, 0.84]))
qtiles = np.array(qtiles)
# shade in std(lnl) region
plt.fill_between(
    np.arange(3, n_bootstraps),
    qtiles[:, 0],
    qtiles[:, 1],
    color="C0",
    alpha=0.5,
    label="1$\sigma$",
)
plt.axhline(lnl, color="k", label="lnl")
plt.xlabel("Number of bootstraps")
plt.ylabel("lnl")
plt.savefig(f"{outdir}/lnl_vs_nbootstraps.png")
