import os
import pickle
from typing import Optional

import numpy as np
from scipy.stats import halfnorm
from sklearn.base import BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import r2_score

from .model import Model

MODEL_SAVE_FILE = "model.pkl"


class SklearnGPModel(Model):
    def __init__(self):
        super().__init__()
        self.kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))

        # # diff between every pair of train_out values
        # err = np.min(np.diff(train_out, axis=0) ** 2)

        # alpha:
        # It can also be interpreted as the variance of additional
        # Gaussian measurement noise on the training observations.

        self._model = GaussianProcessRegressor(
            kernel=self.kernel,
            random_state=0,
            copy_X_train=False,
            n_restarts_optimizer=10,
            alpha=0.1,
            normalize_y=True,
        )

    def train(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        verbose: Optional[bool] = False,
        savedir: Optional[str] = None,
        extra_kwgs={},
    ) -> None:
        """Train the model.

        https://scikit-learn.org/dev/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
        GaussianProcessRegressor(
            kernel=None, *, alpha=1e-10,
            optimizer='fmin_l_bfgs_b',
            n_restarts_optimizer=0,
            normalize_y=False, copy_X_train=True,
            n_targets=None, random_state=None
        )


        """
        (
            train_in,
            test_in,
            train_out,
            test_out,
        ) = self._preprocess_and_split_data(inputs, outputs)

        err = halfnorm.rvs(loc=0, scale=0.5, size=len(train_in))
        self._model = GaussianProcessRegressor(
            kernel=self.kernel,
            random_state=0,
            copy_X_train=False,
            n_restarts_optimizer=10,
            alpha=err,
            normalize_y=True,
        )
        self._model.fit(train_in, train_out)

        return self._post_training(
            (train_in, train_out), (test_in, test_out), savedir, extra_kwgs
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the output of the model for the given input."""
        # x_scaled = self._preprocess_input(x)
        x_scaled = x  # TODO: bug with scaling -- i might be scaling twice
        y_mean, y_std = self._model.predict(x_scaled, return_std=True)
        y_var = y_std**2
        y_lower = y_mean - 1.96 * np.sqrt(y_var)
        y_upper = y_mean + 1.96 * np.sqrt(y_var)
        return y_lower, y_mean, y_upper

    def save(self, savedir: str) -> None:
        """Save a model to a dir."""
        if not self.trained:
            raise ValueError("Model not trained, no point saving")
        os.makedirs(savedir, exist_ok=True)
        with open(f"{savedir}/{MODEL_SAVE_FILE}", "wb") as f:
            pickle.dump([self._model, self.scaler], f)

    @classmethod
    def load(cls, savedir: str) -> "Model":
        """Load a model from a dir."""
        with open(cls.model_fn(savedir), "rb") as f:
            loaded_model = pickle.load(f)
        model = cls()
        model._model = loaded_model[0]
        model.scaler = loaded_model[1]
        model.trained = True
        return model

    @staticmethod
    def saved_model_exists(savedir: str) -> bool:
        """Check if a model exists in the given directory."""
        return os.path.exists(SklearnGPModel.model_fn(savedir))

    @staticmethod
    def model_fn(savedir: str):
        return f"{savedir}/{MODEL_SAVE_FILE}"

    def get_model(self):
        return self._model
