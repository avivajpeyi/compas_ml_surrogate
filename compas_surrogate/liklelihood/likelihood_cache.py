from typing import Dict, Optional

import numpy as np

from compas_surrogate.logger import logger
from compas_surrogate.plotting.corner import plot_corner


class LikelihoodCache(object):
    """Stores the likelihood values for a list of parameters (Optionally stores true model parameters)"""

    def __init__(
        self,
        lnl: np.ndarray,
        params: np.ndarray,
        true_params: Optional[np.array] = None,
    ):
        self.lnl = lnl
        self.params = params
        self.true_params = true_params
        self._clean()

    def _clean(self):
        """Remove likelihoods with nans/inf/-inf"""
        mask = np.isfinite(self.lnl)
        init_len = len(self.lnl)
        self.lnl = self.lnl[mask]
        self.params = self.params[mask]
        new_len = len(self.lnl)
        if new_len != init_len:
            logger.info(
                f"Keeping {new_len}/{init_len} likelihoods (dropping nans/inf)."
            )

    @classmethod
    def from_npz(cls, npz_fn: str):
        """Load likelihood cache from npz file"""
        with np.load(npz_fn) as data:
            lnl = data["lnl"]
            params = data["params"]
            true_params = data.get("true_params", None)
        return cls(lnl, params, true_params)

    @property
    def param_dict(self) -> Dict[str, np.ndarray]:
        return dict(
            aSF=self.params[:, 0],
            bSF=self.params[:, 1],
            cSF=self.params[:, 2],
            dSF=self.params[:, 3],
            muz=self.params[:, 4],
            sigma0=self.params[:, 5],
        )

    @property
    def true_dict(self) -> Dict[str, float]:
        param_names = list(self.param_dict.keys())
        if self.true_params is None:
            return {}
        return dict(zip(param_names, self.true_params))

    @property
    def likelihood(self):
        l = np.exp(self.lnl)
        return l / np.sum(l)

    def save(self, npz_fn: str):
        """Save likelihood cache to npz file"""
        kwgs = dict(lnl=self.lnl, params=self.params)
        if self.true_params is not None:
            kwgs["true_params"] = self.true_params
        np.savez(npz_fn, **kwgs)

    def __repr__(self) -> str:
        return f"LikelihoodCache({len(self.lnl)} likelihoods)"

    def plot(self, fname=""):
        """Plot the samples weighted by their likelihood"""
        labels_to_keep = [
            k for k, v in self.param_dict.items() if len(set(v)) > 1
        ]
        samples = {
            k: v for k, v in self.param_dict.items() if k in labels_to_keep
        }
        if self.true_params is not None:
            true_params = [
                v for k, v in self.true_dict.items() if k in labels_to_keep
            ]

        fig = plot_corner(
            samples, prob=self.likelihood, true_params=true_params
        )
        if fname:
            fig.savefig(fname)
        else:
            return fig

    def sample(self, n_samples: int) -> "LikelihoodCache":
        """Sample from the likelihood distribution"""
        idx = np.random.choice(
            len(self.lnl), size=n_samples, p=self.likelihood
        )
        return LikelihoodCache(
            self.lnl[idx], self.params[idx], self.true_params
        )
