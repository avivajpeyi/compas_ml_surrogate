import datetime
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

from compas_surrogate.logger import logger


class Model(ABC):
    """Base class for surrogate models."""

    def __init__(self):
        self._model = None
        self.trained = False
        self.input_dim = None

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
    ) -> Dict[str, "Metrics"]:
        """Train the model.

        :return Dict[str, Metrics]: metrics for training and testing
        """
        pass

    def fit(self, inputs, outputs):
        """Alias for train."""
        return self.train(inputs, outputs)

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the output of the model for the given input. (lower, mean, upper)"""
        pass

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def prediction_str(self, x: np.ndarray) -> Union[str, List[str]]:
        """Format prediction as a latex string with error bars."""
        lower, mean, upper = self.predict(x)
        q0, q1 = np.abs(np.round(lower - mean, 2)), np.abs(np.round(mean - upper, 2))
        t, b = np.maximum(q0, q1), np.minimum(q0, q1)
        m = np.round(mean, 2)
        strs = [f"{m}^{{+{t}}}_{{-{b}}}" for m, t, b in zip(m, t, b)]
        if len(strs) == 1:
            return strs[0]
        else:
            return strs

    def _preprocess_and_split_data(self, input, output, test_size=0.2):
        """
        Preprocess and split data into training and testing sets.
        :param input:
        :param output:
        :param test_size:
        :return: (train_in, test_in, train_out, test_out)
        """

        # check shape of input and output are the same
        if input.shape[0] != output.shape[0]:
            raise ValueError(
                f"Input ({input.shape}) and output ({output.shape}) must have the same number of samples"
            )

        # check that input and output are tensors (len(shape) > 1)
        if len(input.shape) < 2 or len(output.shape) < 2:
            raise ValueError("Input and output must be tensors (len(shape) > 1)")

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
        """Return the average of absolute errors of all the data points in the given dataset.
        np.mean(np.abs(pred_y - data[1]))
        """
        true_y = data[1]
        _, pred_y, _ = self(data[0])
        return round(metrics.mean_absolute_error(true_y, pred_y), 2)

    def mean_squared_error(self, data):
        """Return the average of the squares of the errors of all the data points in the given dataset.
        np.mean((pred_y - data[1]) ** 2)
        """
        true_y = data[1]
        _, pred_y, _ = self(data[0])
        return round(metrics.mean_squared_error(true_y, pred_y), 2)

    def r_sqred(self, data):
        """Return the R^2 =  1- relative mean squared error of the model
        This is the coefficient of determination.
        This tells us how well the unknown samples will be predicted by our model.
        The best possible score is 1.0, but the score can be negative as well.
        1 - np.mean((pred_y - true_y) ** 2) / np.var(true_y)
        """
        true_y = data[1]
        _, pred_y, _ = self(data[0])
        return round(metrics.r2_score(true_y, pred_y), 2)

    def _get_metrics(self, data):
        return Metrics(
            self.mean_absolute_error(data),
            self.mean_squared_error(data),
            self.r_sqred(data),
        )

    def train_test_metrics(self, train_data, test_data):
        return {
            "train": self._get_metrics(train_data),
            "test": self._get_metrics(test_data),
        }


class Metrics:
    def __init__(self, mae, mse, r2):
        self.mae = mae
        self.mse = mse
        self.r2 = r2

    def __str__(self):
        return f"MAE: {self.mae}, MSE: {self.mse}, R2: {self.r2}"

    def __dict__(self):
        return {"MAE": self.mae, "MSE": self.mse, "R2": self.r2}

    def __repr__(self):
        return self.__str__()
