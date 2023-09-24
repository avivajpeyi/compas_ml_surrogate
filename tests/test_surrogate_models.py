import os
import unittest
from glob import glob
from typing import Callable, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy

np.random.seed(1)

TEAR_DOWN = False

from compas_surrogate.surrogate.models import SklearnGPModel

CURVY_F = (
    lambda x: 0.2 + 0.4 * x**2 + 0.3 * x * np.sin(15 * x) + 0.05 * np.cos(50 * x)
)

WAVELET_F = lambda x: scipy.stats.norm(0.5, 0.15).pdf(x) * np.sin(50 * x)


def train_and_save_model(model_class, data: np.ndarray, model_path: str):
    model = model_class()
    model.train(data[0], data[1], verbose=True, savedir=model_path)
    preds = model.predict(data[0])
    loaded_model = model_class.load(model_path)
    loaded_preds = loaded_model.predict(data[0])
    assert np.allclose(preds, loaded_preds)


def plot(ax, true_f, models, model_names):
    data = generate_data(true_f, 500)
    x = data[0]
    y = true_f(x)
    ax.plot(x, y, label="True", color="black", ls="--", zorder=10)
    for i, model in enumerate(models):
        low_y, mean_y, up_y = model(x)
        ax.plot(
            x,
            mean_y,
            label=f"{model_names[i]} (R^2:{model.r_sqred(data):.2f})",
            color=f"C{i}",
            alpha=0.5,
        )
        ax.fill_between(
            x.flatten(),
            low_y.flatten(),
            up_y.flatten(),
            alpha=0.1,
            color=f"C{i}",
        )
    ax.legend(frameon=False)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.axis("off")


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
        from compas_surrogate.surrogate.models.deep_gp_model import DeepGPModel

        self.gp_model_tester(DeepGPModel)

    def test_sklearn_gp_model(self):
        self.gp_model_tester(SklearnGPModel)

    def gp_model_tester(self, gp_model_class):
        gp_name = gp_model_class.__name__
        pts = [50]
        test_funcs = [CURVY_F]
        num_f = len(test_funcs)

        fig, ax = plt.subplots(num_f, 1, figsize=(5, 3 * num_f))
        if num_f == 1:
            ax = [ax]

        for fi, f in enumerate(test_funcs):
            paths = [f"{self.outdir}/{gp_name}_{fi}_model_n{i}" for i in pts]
            model_names = [f"{i} Training pts " for i in pts]
            for path, pt in zip(paths, pts):
                train_and_save_model(gp_model_class, generate_data(f, pt), path)
            plot(
                ax[fi],
                true_f=f,
                models=[gp_model_class.load(p) for p in paths],
                model_names=model_names,
            )

        plt.tight_layout()
        plt.savefig(f"{self.outdir}/{gp_name}.png")
