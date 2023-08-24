import numpy as np
from bilby.core.likelihood import Likelihood

# from .models.model import Model


class SurrogateLikelihood(Likelihood):
    def __init__(self, lnl_surrogate: "Model", parameter_keys: list):
        super().__init__({k: 0 for k in parameter_keys})
        self.param_keys = parameter_keys
        self.surr = lnl_surrogate

    def log_likelihood(self):
        params = np.array([[self.parameters[k] for k in self.param_keys]])
        assert params.shape == (
            1,
            len(self.param_keys),
        ), f"Incorrect shape of parameters: {params.shape}"
        y_lower, y_mean, y_upper = self.surr(params)
        return y_mean.ravel()[0]
