import matplotlib.pyplot as plt
import numpy as np

from compas_surrogate.plotting.image_utils import horizontal_concat

from .model_corner import plot_model_corner

MODEL_COL = "tab:green"
TRAIN_COL = "tab:blue"
TEST_COL = "tab:red"
TRUE_COL = "tab:orange"


def plot_model_diagnostics(model, train_data, test_data, savedir: str = None, kwgs={}):
    """Plot the training results."""

    kwgs["model_col"] = kwgs.get("model_col", MODEL_COL)
    kwgs["train_col"] = kwgs.get("train_col", TRAIN_COL)
    kwgs["test_col"] = kwgs.get("test_col", TEST_COL)

    fname1 = f"{savedir}/model_diagnostic_ppc.png"
    plot_model_predictive_check(model, train_data, test_data, fname1, kwgs)
    fname2 = f"{savedir}/model_diagnostic_err.png"
    plot_predicted_vs_true(model, train_data, test_data, fname2, kwgs)
    fname3 = f"{savedir}/model_diagnostic_hist.png"
    plot_error_hist(model, train_data, test_data, fname3, kwgs)

    horizontal_concat(
        [fname1, fname2, fname3], f"{savedir}/model_diagnostic.png", rm_orig=True
    )


def plot_error_hist(model, train_data, test_data, fname, kwgs):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    __plot_error_hist(
        model, ax, train_data, {"color": kwgs["train_col"], "label": "Train"}
    )
    __plot_error_hist(
        model, ax, test_data, {"color": kwgs["test_col"], "label": "Test"}
    )
    ax.legend()
    ax.set_title("Error Histogram")
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(fname, dpi=500)
    plt.close(fig)


def plot_predicted_vs_true(model, train_data, test_data, fname, kwgs):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    __plot_prediction_comparison(
        model, ax, train_data, {"color": kwgs["train_col"], "label": "Train"}
    )
    datarange = [train_data[1].min(), train_data[1].max()]
    # extend the range by 1% on either side
    datarange[0] -= 0.01 * np.abs(datarange[0])
    datarange[1] += 0.01 * np.abs(datarange[1])
    ax.plot(
        datarange,
        datarange,
        "k--",
        lw=0.3,
        zorder=10,
    )
    __plot_prediction_comparison(
        model, ax, test_data, {"color": kwgs["test_col"], "label": "Test"}
    )
    ax.legend()
    ax.set_title("Prediction vs True")
    fig.tight_layout()
    fig.savefig(fname, dpi=500)
    plt.close(fig)


def plot_model_predictive_check(model, train_data, test_data, fname, kwgs):
    if model.input_dim == 1:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        __plot_1d_model(model, ax, (train_data[0], train_data[1]), kwgs)
        ax.plot(
            test_data[0],
            test_data[1],
            "o",
            color=kwgs["test_col"],
            label="Test",
        )
        if "labels" in kwgs:
            ax.legend(kwgs["labels"][0])
        if "truths" in kwgs:
            ax.vlines(kwgs["truths"][0], 0, 1, color="tab:orange", linestyle="--")
        ax.legend()
        ax.set_title("Predictive Check")
        fig.tight_layout()
    else:
        _, pred, _ = model(train_data[0])
        fig = plot_model_corner(train_data, test_data, pred, kwgs)
    fig.savefig(fname, dpi=500)
    plt.close(fig)


def __plot_prediction_comparison(model, ax, data, kwgs):
    """Plot the prediction vs the true values."""
    color = kwgs.get("color", "tab:blue")
    label = kwgs.get("label", "Prediction")
    r2 = model.r_sqred(data)
    label = f"{label} (R2: {r2})"
    true_y = data[1].flatten()
    pred_low, pred_y, pred_up = model(data[0])
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


def __plot_1d_model(model, ax, train_data, kwgs):
    """Plot the model in 1D."""
    xlin = np.linspace(train_data[0].min(), train_data[0].max(), 100)
    pred_low, pred_y, pred_up = model(xlin.reshape(-1, 1))
    model_col = kwgs.get("model_col", MODEL_COL)
    data_col = kwgs.get("data_col", TRAIN_COL)
    ax.fill_between(xlin, pred_low, pred_up, color=model_col, alpha=0.2)
    ax.plot(xlin, pred_y, color=model_col, label="Model")
    ax.plot(train_data[0], train_data[1], "o", color=data_col, label="Training Data")
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")


def __plot_error_hist(model, ax, data, kwgs):
    color = kwgs.get("color", "tab:blue")
    label = kwgs.get("label", "Prediction")
    true_y = data[1].flatten()
    _, pred_y, _ = model(data[0])
    ax.hist(
        pred_y - true_y,
        bins=50,
        color=color,
        label=label,
        alpha=0.5,
        density=True,
        histtype="step",
        lw=2,
    )
    ax.set_xlabel("True - Pred")
