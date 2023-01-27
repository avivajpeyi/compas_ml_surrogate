import datetime
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split

from compas_surrogate.logger import logger


class Model(ABC):
    """Base class for surrogate models."""

    @classmethod
    @abstractmethod
    def load(cls, savedir: str) -> "Model":
        """Load a model from a dir."""
        pass

    @abstractmethod
    def save(self, savedir: str) -> None:
        """Save a model to a dir."""
        pass

    @abstractmethod
    def train(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        verbose: Optional[bool] = False,
    ) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the output of the model for the given input."""
        pass

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def _preprocess_and_split_data(self, input, output, test_size=0.2):

        # check shape of input and output are the same
        if input.shape[0] != output.shape[0]:
            raise ValueError(
                "Input and output must have the same number of samples"
            )

        # check that input and output are tensors (len(shape) > 1)
        if len(input.shape) < 2 or len(output.shape) < 2:
            raise ValueError(
                "Input and output must be tensors (len(shape) > 1)"
            )

        (train_in, test_in, train_out, test_out) = train_test_split(
            input, output, test_size=test_size
        )

        logger.info(
            f"Training surrogate In({train_in.shape})-->Out({train_out.shape}) [testing:{len(test_out)}]"
        )

        return train_in, test_in, train_out, test_out

    @staticmethod
    def current_time():
        return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def mean_absolute_error(self, data):
        """Return the mean absolute error of the model."""
        _, pred_y, _ = self(data[0])
        return np.mean(np.abs(pred_y - data[1]))

    def mean_squared_error(self, data):
        _, pred_y, _ = self(data[0])
        return np.mean((pred_y - data[1]) ** 2)

    def r_sqred(self, data):
        """Return the R^2 =  1- relative mean squared error of the model"""
        _, pred_y, _ = self(data[0])
        true_y = data[1]
        return 1 - np.mean((pred_y - true_y) ** 2) / np.var(true_y)
