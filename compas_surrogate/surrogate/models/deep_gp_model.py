from typing import Callable, Tuple, Union

import gpflow
import numpy as np
import tensorflow as tf

from .model import Model


class DeepGPModel(Model):
    def __init__(self):
        self._model = None  # the model
        self.trained = False
        self.input_dim = None

    def train(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
        """Train the model."""
        self._model = gpflow.models.GPR(
            data=(inputs, outputs),
            kernel=gpflow.kernels.SquaredExponential(),
        )
        opt = gpflow.optimizers.Scipy()
        opt.minimize(
            self._model.training_loss, self._model.trainable_variables
        )
        self.trained = True
        self.input_dim = inputs.shape[1]

        self._model.predict = tf.function(
            lambda Xnew: self._model.predict_y(Xnew),
            input_signature=[
                tf.TensorSpec(shape=[None, self.input_dim], dtype=tf.float64)
            ],
        )

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray]:
        """Predict the output of the model for the given input."""
        if not self.trained:
            raise RuntimeError("Model not trained yet.")

        y_mean, y_var = self._model.predict(x)
        y_lower = y_mean - 1.96 * np.sqrt(y_var)
        y_upper = y_mean + 1.96 * np.sqrt(y_var)
        return y_lower.numpy(), y_mean.numpy(), y_upper.numpy()

    def save(self, savedir: str) -> None:
        """Save a model to a file."""
        if not self.trained:
            raise RuntimeError("Model not trained yet.")
        tf.saved_model.save(self._model, savedir)

    @classmethod
    def load(cls, savedir: str) -> "Model":
        """Load a model from a dir."""
        model = cls()
        model._model = tf.saved_model.load(savedir)
        model.trained = True
        return model
