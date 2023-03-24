import datetime
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from corner import corner
from sklearn import metrics
from sklearn.model_selection import train_test_split

from compas_surrogate.logger import logger
from compas_surrogate.plotting.corner import plot_corner
from compas_surrogate.plotting.image_utils import horizontal_concat

from .utils import plot_model_corner


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
        savedir: Optional[str] = None,
    ) -> Dict[str, "Metrics"]:
        """Train the model.

        :return Dict[str, Metrics]: metrics for training and testing
        """
        pass

    def _post_training(self, training_data, testing_data, savedir):
        """Post training processing."""
        self.trained = True
        self.input_dim = training_data[0].shape[1]
        if savedir:
            self.save(savedir)
            self.plot_diagnostics(training_data, testing_data, savedir)
        return self.train_test_metrics(training_data, testing_data)

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

    def plot_diagnostics(self, train_data, test_data, savedir: str = None):
        """Plot the training results."""

        kwgs = dict(
            model_col="tab:green",
            train_col="tab:blue",
            test_col="tab:orange",
        )

        fname1 = f"{savedir}/model_diagnostic_ppc.png"
        self.plot_model_predictive_check(train_data, test_data, fname1, kwgs)
        fname2 = f"{savedir}/model_diagnostic_err.png"
        self.plot_predicted_vs_true(train_data, test_data, fname2, kwgs)
        horizontal_concat([fname1, fname2], f"{savedir}/model_diagnostic.png")
        # os.remove(fname1)
        # os.remove(fname2)

    def plot_predicted_vs_true(self, train_data, test_data, fname, kwgs):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        self._plot_prediction_comparison(
            ax, train_data, {"color": kwgs["train_col"], "label": "Train"}
        )
        datarange = [train_data[1].min(), train_data[1].max()]
        ax.plot(
            datarange,
            datarange,
            "k--",
            lw=0.1,
            zorder=-10,
        )
        self._plot_prediction_comparison(
            ax, test_data, {"color": kwgs["test_col"], "label": "Test"}
        )
        ax.legend()
        ax.set_title("Prediction vs True")
        fig.tight_layout()
        fig.savefig(fname)

    def plot_model_predictive_check(self, train_data, test_data, fname, kwgs):
        if self.input_dim == 1:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            self._plot_1d_model(ax, (train_data[0], train_data[1]), kwgs)
            ax.plot(
                test_data[0],
                test_data[1],
                "o",
                color=kwgs["test_col"],
                label="Test",
            )
            ax.legend()
            ax.set_title("Predictive Check")
            fig.tight_layout()
        else:
            _, pred, _ = self(train_data[0])
            fig = plot_model_corner(train_data, test_data, pred, kwgs)

        fig.savefig(fname, dpi=500)

    def _plot_prediction_comparison(self, ax, data, kwgs):
        """Plot the prediction vs the true values."""
        color = kwgs.get("color", "tab:blue")
        label = kwgs.get("label", "Prediction")
        r2 = self.r_sqred(data)
        label = f"{label} (R2: {r2})"
        true_y = data[1].flatten()
        pred_low, pred_y, pred_up = self(data[0])
        ax.errorbar(
            true_y,
            pred_y,
            marker="o",
            linestyle="None",
            yerr=[pred_y - pred_low, pred_up - pred_y],
            color=color,
            label=label,
            markersize=1,
            alpha=0.5,
        )
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")

    def _plot_1d_model(self, ax, train_data, kwgs):
        """Plot the model in 1D."""
        xlin = np.linspace(train_data[0].min(), train_data[0].max(), 100)
        pred_low, pred_y, pred_up = self(xlin.reshape(-1, 1))
        model_col, data_col = kwgs.get("model_col", "tab:green"), kwgs.get(
            "data_col", "tab:blue"
        )
        ax.fill_between(xlin, pred_low, pred_up, color=model_col, alpha=0.2)
        ax.plot(xlin, pred_y, color=model_col, label="Model")
        ax.plot(
            train_data[0], train_data[1], "o", color=data_col, label="Training Data"
        )
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")


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
