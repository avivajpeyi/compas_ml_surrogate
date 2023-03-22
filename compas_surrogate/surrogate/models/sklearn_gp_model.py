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
        kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
        self._model = GaussianProcessRegressor(
            kernel=kernel, random_state=0
        )  #  copy_X_train=False, normalize_y=True)
        self.trained = False
        self.input_dim = None

    def train(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        verbose: Optional[bool] = False,
    ) -> None:
        """Train the model."""
        (
            train_in,
            test_in,
            train_out,
            test_out,
        ) = self._preprocess_and_split_data(inputs, outputs)
        self._model.fit(train_in, train_out)
        self.trained = True
        self.input_dim = inputs.shape[1]
        metrics = self.train_test_metrics([train_in, train_out], [test_in, test_out])
        return metrics

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
        loaded_model = pickle.load(open(filename, "rb"))
        model = cls()
        model._model = loaded_model
        model.trained = True
        return model

    def get_model(self):
        return self._model
