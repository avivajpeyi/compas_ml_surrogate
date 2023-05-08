import os
import pickle
from typing import Optional

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from .model import Model

MODEL_SAVE_FILE = "model.pkl"


class SklearnGPModel(Model):
    def __init__(self):
        self._model = None
        self.trained = False
        self.input_dim = None
        kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))

        # # diff between every pair of train_out values
        # err = np.min(np.diff(train_out, axis=0) ** 2)

        self._model = GaussianProcessRegressor(
            kernel=kernel,
            random_state=0,
            copy_X_train=False,
            n_restarts_optimizer=10,
            alpha=10,
        )

    def train(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        verbose: Optional[bool] = False,
        savedir: Optional[str] = None,
        extra_kwgs={},
    ) -> None:
        """Train the model."""
        (
            train_in,
            test_in,
            train_out,
            test_out,
        ) = self._preprocess_and_split_data(inputs, outputs)
        kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))

        # diff between every pair of train_out values
        err = np.min(np.diff(train_out, axis=0) ** 2)

        self._model = GaussianProcessRegressor(
            kernel=kernel,
            random_state=0,
            copy_X_train=False,
            n_restarts_optimizer=10,
            alpha=10,
        )
        self._model.fit(train_in, train_out)

        return self._post_training(
            (train_in, train_out), (test_in, test_out), savedir, extra_kwgs
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the output of the model for the given input."""
        y_mean, y_std = self._model.predict(x, return_std=True)
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
            pickle.dump(self._model, f)

    @classmethod
    def load(cls, savedir: str) -> "Model":
        """Load a model from a dir."""
        filename = f"{savedir}/{MODEL_SAVE_FILE}"
        with open(filename, "rb") as f:
            loaded_model = pickle.load(f)
        model = cls()
        model._model = loaded_model
        model.trained = True
        return model

    def get_model(self):
        return self._model
