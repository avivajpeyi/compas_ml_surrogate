from collections import namedtuple
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compas_surrogate.logger import logger
from compas_surrogate.plotting import safe_savefig
from compas_surrogate.plotting.corner import plot_corner


class LikelihoodCache(object):
    """Stores the likelihood values for a list of parameters (Optionally stores true model parameters)"""

    def __init__(
        self,
        lnl: np.ndarray,
        params: np.ndarray,
        true_params: Optional[np.array] = None,
        true_lnl: Optional[float] = None,
    ):
        self.lnl = lnl
        self.params = params
        self.true_params = true_params
        self.true_lnl = true_lnl
        self._clean()

    @property
    def n(self) -> int:
        return len(self.lnl)

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

    def get_varying_params(self, ret_dict=False) -> Union[Dict, np.ndarray]:
        """Get the parameters that change"""
        params = {k: v for k, v in self.param_dict.items() if not len(set(v)) == 1}
        if ret_dict:
            return params
        else:
            return np.array(list(params.values()))

    def get_varying_param_keys(self) -> List[str]:
        """Get the names for the parameters that change"""
        data = self.get_varying_params(ret_dict=True)
        return list(data.keys())

    @classmethod
    def from_npz(cls, npz_fn: str):
        """Load likelihood cache from npz file"""
        with np.load(npz_fn) as data:
            lnl = data["lnl"]
            params = data["params"]
            true_params = data.get("true_params", None)
            true_lnl = data.get("true_lnl", None)
        return cls(lnl, params, true_params, true_lnl)

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

    def get_true_param_lnl(self) -> float:
        """Get the likelihood of the true parameters"""
        if self.true_lnl is None:
            # get lnl value closest to true params
            idx = np.argmin(np.sum(np.abs(self.params - self.true_params), axis=1))
            self.true_lnl = self.lnl[idx]
        return self.true_lnl

    @property
    def normed_likelihood(self):
        return np.exp(self.lnl - np.max(self.lnl))

    def save(self, npz_fn: str):
        """Save likelihood cache to npz file"""
        kwgs = dict(lnl=self.lnl, params=self.params)
        if self.true_params is not None:
            kwgs["true_params"] = self.true_params
        if self.true_lnl is not None:
            kwgs["true_lnl"] = self.true_lnl
        np.savez(npz_fn, **kwgs)

    def __repr__(self) -> str:
        return f"LikelihoodCache({len(self.lnl)} likelihoods)"

    def plot(self, fname="", show_datapoints=False):
        """Plot the samples weighted by their likelihood"""

        if len(self.lnl) == 0:
            logger.error("No likelihoods to plot")
            return

        samples = self.get_varying_params(ret_dict=True)
        true_params = None
        if self.true_params is not None:
            true_params = [self.true_dict[k] for k in list(samples.keys())]

        fig = plot_corner(
            samples,
            prob=self.normed_likelihood,
            true_params=true_params,
            show_datapoints=show_datapoints,
        )
        fig.suptitle(
            f"Likelihood weighted {self.n:,} samples\n(True lnl: {self.true_lnl:.2f})"
        )
        if fname:
            safe_savefig(fig, fname)
            plt.close(fig)
        else:
            return fig

    def sample(self, n_samples: int) -> "LikelihoodCache":
        """Sample from the likelihood distribution"""
        p = self.normed_likelihood / np.sum(self.normed_likelihood)
        idx = np.random.choice(len(self.lnl), size=n_samples, p=p)
        return LikelihoodCache(
            self.lnl[idx], self.params[idx], self.true_params, self.true_lnl
        )

    @property
    def true_param_vals(self):
        data = np.array([self.true_dict[k] for k in self.get_varying_param_keys()])
        return data.reshape(1, -1)

    @property
    def dataframe(self) -> pd.DataFrame:
        data = self.get_varying_params(ret_dict=True)
        df = pd.DataFrame(data)
        df["lnl"] = self.lnl
        return df

    def max_lnl_param_dict(self) -> Dict[str, float]:
        """Get the param corresponding to the maximum likelihood event"""
        df = pd.DataFrame({**self.param_dict, "lnl": self.lnl})
        max_lnl_row = df.loc[df["lnl"].idxmax()].to_dict()
        del max_lnl_row["lnl"]
        return max_lnl_row

    def get_sample_histograms(
        self, bins=None
    ) -> Dict[str, namedtuple("Hist", ["hist", "bin_edges"])]:
        df = self.dataframe
        Hist = namedtuple("Hist", ["hist", "bin_edges"])
        hists = {}
        if bins is None:
            bins = {k: int(np.sqrt(len(df))) for k in self.get_varying_param_keys()}
        for k in self.get_varying_param_keys():
            hist, bin_edges = np.histogram(df[k], bins=bins[k], density=True)
            hists[k] = Hist(hist, bin_edges)
        return hists
