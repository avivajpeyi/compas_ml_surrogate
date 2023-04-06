import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bilby.core.prior import Gaussian, PriorDict
from sklearn.datasets import make_moons

np.random.seed(0)

# 2d banana distribution
data = {"x": [], "y": []}

n = 100000
u1 = np.random.normal(0, 1, n)
u2 = np.random.normal(0, 1, n)
a = 1.15
b = 0.5
x = u1 * a
y = (u2 / a) + b * (u1**2 + a**2)

count2d, x_bins, y_bins = np.histogram2d(
    x, y, bins=50, range=[[-5, 5], [-2, 6]], density=True
)

xmid = (x_bins[1:] + x_bins[:-1]) / 2
ymid = (y_bins[1:] + y_bins[:-1]) / 2
xx, yy = np.meshgrid(xmid, ymid)
zz = count2d.T

df = pd.DataFrame(dict(x=xx.flatten(), y=yy.flatten(), z=count2d.flatten()))

fig, ax = plt.subplots(figsize=(3, 3))
ax.pcolormesh(x_bins, y_bins, count2d.T, cmap="hot")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.tight_layout()

samples = df.sample(n=100, weights=df.z, replace=True, random_state=0)
ax.scatter(
    samples.x,
    samples.y,
    marker="*",
    color="dodgerblue",
    s=25,
    alpha=0.5,
    label="Sample w/ weights",
)

## use emcee to sample from the distribution
import emcee
from scipy.stats import norm


def log_likelihood(theta, x, y):
    mu, sigma = theta
    model = norm.pdf(x, mu, sigma)
    return np.sum(np.log(model))


def log_prior(theta):
    mu, sigma = theta
    if -5 < mu < 5 and 0 < sigma < 5:
        return 0.1
    return -np.inf


# start sampler
ndim, nwalkers = 2, 100
pos = [np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_likelihood, args=(samples.x, samples.y)
)

for i in sampler.sample(pos, iterations=1000, progress=True):
    print(i)

samples = sampler.get_chain(flat=True)
ax.scatter(
    samples[:, 0],
    samples[:, 1],
    marker=".",
    color="limegreen",
    s=1,
    alpha=0.5,
    label="Sample w/ emcee",
)

plt.legend(fontsize=8)
plt.tight_layout()
plt.show()
