from typing import Callable, Optional, Tuple, Union

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.monitor import ScalarToTensorBoard as ScalarCallback

# scikit-learn train-test split
from compas_surrogate.logger import logger

from .model import Model


class DeepGPModel(Model):
    def __init__(self):
        self._model = None  # the model
        self.trained = False
        self.input_dim = None

    def train(
        self,
        inputs: np.ndarray,
        outputs: np.ndarray,
        verbose: Optional[bool] = False,
        log_dir: Optional[str] = "training_logs",
    ) -> None:
        """Train the model."""

        (
            train_in,
            test_in,
            train_out,
            test_out,
        ) = self._preprocess_and_split_data(inputs, outputs)

        self._model = gpflow.models.GPR(
            data=(train_in, train_out),
            kernel=gpflow.kernels.SquaredExponential(),
        )
        monitor = None
        if verbose:
            logdir = log_dir + "/" + self.current_time()

            train_d, test_d = (train_in, train_out), (test_in, test_out)

            test_r2 = lambda: self.r_sqred(test_d)
            train_r2 = lambda: self.r_sqred(train_d)
            print_stats = lambda: self._print_training_stats(train_d, test_d)

            tasks = []
            tasks += [gpflow.monitor.ModelToTensorBoard(logdir, self._model)]
            tasks += [
                ScalarCallback(logdir, self._model.training_loss, "loss")
            ]
            tasks += [ScalarCallback(logdir, train_r2, "train_R2")]
            tasks += [ScalarCallback(logdir, test_r2, "test_R2")]
            tasks += [gpflow.monitor.ExecuteCallback(print_stats)]
            task_group = gpflow.monitor.MonitorTaskGroup(tasks, period=1)
            monitor = gpflow.monitor.Monitor(task_group)

        opt = gpflow.optimizers.Scipy()
        opt.minimize(
            self._model.training_loss,
            self._model.trainable_variables,
            step_callback=monitor,
        )
        self.trained = True
        self.input_dim = inputs.shape[1]

        self._model.predict = tf.function(
            lambda xnew: self.__train_m_pred(xnew),
            input_signature=[
                tf.TensorSpec(shape=[None, self.input_dim], dtype=tf.float64)
            ],
        )

    def __train_m_pred(self, xnew):
        """Helper function for the predict method while training"""
        return self._model.predict_f(xnew, full_cov=False)

    def predict(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict the output of the model for the given input."""
        if hasattr(self._model, "predict"):
            pred = self._model.predict
        else:
            pred = self.__train_m_pred

        y_mean, y_var = pred(x)
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

    # non-abstract methods

    def _print_training_stats(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        test_data: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        train_l = self._model.training_loss()
        train_r2 = self.r_sqred(train_data)
        test_r2 = self.r_sqred(test_data)

        er_st = "{}: <{:.2f}|{:.2f}>"

        log = (
            f"Loss:{train_l:.2f} | "
            f"{er_st.format('R^2', train_r2, test_r2)} "
        )
        logger.info(log)
