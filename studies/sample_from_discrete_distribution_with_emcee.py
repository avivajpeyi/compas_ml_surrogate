import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bilby.core.prior import Gaussian, PriorDict

from compas_surrogate.plotting.image_utils import horizontal_concat

np.random.seed(0)


class WalkerRandomSampling(object):
    """Walker's alias method for random objects with different probablities.

    Based on the implementation of Denis Bzowy at the following URL:
    http://code.activestate.com/recipes/576564-walkers-alias-method-for-random-objects-with-diffe/
    """

    def __init__(self, weights, keys=None):
        """Builds the Walker tables ``prob`` and ``inx`` for calls to `random()`.
        The weights (a list or tuple or iterable) can be in any order and they
        do not even have to sum to 1."""
        n = self.n = len(weights)
        if keys is None:
            self.keys = keys
        else:
            self.keys = np.array(keys)

        if isinstance(weights, (list, tuple)):
            weights = np.array(weights, dtype=float)
        elif isinstance(weights, np.ndarray):
            if weights.dtype != float:
                weights = weights.astype(float)
        else:
            weights = np.array(list(weights), dtype=float)

        if weights.ndim != 1:
            raise ValueError("weights must be a vector")

        weights = weights * n / weights.sum()

        inx = -np.ones(n, dtype=int)
        short = np.where(weights < 1)[0].tolist()
        long = np.where(weights > 1)[0].tolist()
        while short and long:
            j = short.pop()
            k = long[-1]

            inx[j] = k
            weights[k] -= 1 - weights[j]
            if weights[k] < 1:
                short.append(k)
                long.pop()

        self.prob = weights
        self.inx = inx

    def random(self, count=None):
        """Returns a given number of random integers or keys, with probabilities
        being proportional to the weights supplied in the constructor.

        When `count` is ``None``, returns a single integer or key, otherwise
        returns a NumPy np.array with a length given in `count`.
        """
        if count is None:
            u = np.random.random()
            j = np.random.randint(self.n)
            k = j if u <= self.prob[j] else self.inx[j]
            return self.keys[k] if self.keys is not None else k

        u = np.random.random(count)
        j = np.random.randint(self.n, size=count)
        k = np.where(u <= self.prob[j], j, self.inx[j])
        return self.keys[k] if self.keys is not None else k


#
#
#
# 2d banana distribution
data = {"x": [], "y": []}

n = 1000000
u1 = np.random.normal(0, 1, n)
u2 = np.random.normal(0, 1, n)
a = 1.15
b = 0.5
x = u1 * a
y = (u2 / a) + b * (u1**2 + a**2)

count2d, x_bins, y_bins = np.histogram2d(
    x, y, bins=25, range=[[-5, 5], [-4, 6]], density=True
)

xmid = (x_bins[1:] + x_bins[:-1]) / 2
ymid = (y_bins[1:] + y_bins[:-1]) / 2
xx, yy = np.meshgrid(xmid, ymid)
zz = count2d

df = pd.DataFrame(dict(x=xx.flatten(), y=yy.flatten(), z=count2d.flatten()))

fig, axes = plt.subplots(3, 1, figsize=(3, 9))
ax = axes[0]
cbar = ax.pcolormesh(x_bins, y_bins, count2d.T, cmap="hot", label="Raw Hist")
fig.colorbar(cbar, ax=ax)
plt.tight_layout()

N = 1000
samples = df.sample(n=N, weights=df.z, replace=True, random_state=0)
hist, _, _ = np.histogram2d(x=samples.x, y=samples.y, bins=[x_bins, y_bins])
ax = axes[1]
cbar = ax.pcolormesh(x_bins, y_bins, hist, cmap="hot", label="pandas sample")
fig.colorbar(cbar, ax=ax)


keys = df[["x", "y"]].values
wrand = WalkerRandomSampling(weights=df.z.values, keys=keys)
samples = wrand.random(N)
hist, _, _ = np.histogram2d(x=samples[:, 0], y=samples[:, 1], bins=[x_bins, y_bins])
ax = axes[2]
cbar = ax.pcolormesh(x_bins, y_bins, hist, cmap="hot")
fig.colorbar(cbar, ax=ax)

plt.legend(fontsize=8)
plt.tight_layout()
plt.show()
#
# import seaborn as sns
#
# hist_2d = pd.DataFrame(np.array([[129, 162, 178, 182, 182, 182],
#                    [122, 163, 185, 191, 189, 185],
#                    [115, 164, 192, 200, 197, 189],
#                    [ 94, 136, 158, 162, 157, 152],
#                    [ 74, 108, 124, 125, 118, 116],
#                    [ 53,  80,  90,  88,  79,  80]]),
#                   index=range(2,8), columns=range(8,14))
# sns.heatmap(hist_2d)
# plt.savefig('v1.png')
# plt.close()
#
# keys = list(itertools.product(hist_2d.index, hist_2d.columns))
# values = hist_2d.values.flatten()
#
# wrand = WalkerRandomSampling(weights=values, keys=keys)
# samples = wrand.random(100000)
#
# hist,_,_ = np.histogram2d(x= samples[:,0], y=samples[:,1], bins=6)
# sns.heatmap(hist)
# plt.savefig('v2.png')
#
# horizontal_concat(["v1.png", "v2.png"], 'plts.png')
