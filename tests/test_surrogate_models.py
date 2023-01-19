import os
import unittest
from glob import glob
from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np

TEAR_DOWN = False

from compas_surrogate.surrogate.models import DeepGPModel, GPModel

CURVY_F = (
    lambda x: 0.2
    + 0.4 * x**2
    + 0.3 * x * np.sin(15 * x)
    + 0.05 * np.cos(50 * x)
)


def train_and_save_model(model_class, data: np.ndarray, model_path: str):
    model = model_class()
    model.train(data[0], data[1])
    preds = model.predict(data[0])
    model.save(model_path)
    loaded_model = model_class.load(model_path)
    loaded_preds = loaded_model.predict(data[0])
    assert np.allclose(preds, loaded_preds)


def plot(true_f, model_fs, fname):
    data = generate_data(true_f, 500)
    x = data[0]
    plt.plot(x, true_f(x), label="True", color="black", ls="--", zorder=10)
    for i, model_f in enumerate(model_fs):
        low_y, mean_y, up_y = model_f(x)
        plt.plot(x, mean_y, label=f"Model {i}", color=f"C{i}", alpha=0.5)
        plt.fill_between(
            x.flatten(),
            low_y.flatten(),
            up_y.flatten(),
            alpha=0.1,
            color=f"C{i}",
        )
    plt.legend()
    plt.xlim(x.min(), x.max())
    plt.savefig(fname)


def generate_data(func: Callable, n=50) -> np.ndarray:
    x = np.random.uniform(0, 1, n)
    x = np.sort(x)
    y = func(x)
    X = [[i] for i in x]
    Y = [[i] for i in y]
    return np.array([X, Y])


class TestSurrogate(unittest.TestCase):
    def setUp(self) -> None:
        self.outdir = "out_surrogate"
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir, exist_ok=True)

    def tearDown(self) -> None:
        if TEAR_DOWN is False:
            return
        for f in glob(os.path.join(self.outdir, "*")):
            os.remove(f)
        os.rmdir(self.outdir)

    def test_deep_gp_model(self):
        pts = [15, 50, 200]
        paths = [f"{self.outdir}/deep_gp_model_{i}" for i in pts]
        for path, pt in zip(paths, pts):
            train_and_save_model(DeepGPModel, generate_data(CURVY_F, pt), path)
        plot(
            CURVY_F,
            [DeepGPModel.load(p).predict for p in paths],
            f"{self.outdir}/deep_gp_model.png",
        )
