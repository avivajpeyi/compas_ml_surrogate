import matplotlib.pyplot as plt
import numpy as np
from compas_surrogate.cosmic_integration.universe import Universe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from matplotlib.patches import Rectangle
from scipy.interpolate import griddata
from compas_surrogate.data_generation.likelihood_cacher import get_training_lnl_cache
from compas_surrogate.plotting.image_utils import horizontal_concat, vertical_concat
from compas_surrogate.plotting.gif_generator import make_gif

IDX = [i for i in range(0, 1000, 25)]


def plot_1d(df, true_val):
    lnl = df["lnl"]
    # get other column name
    data_label = [col for col in df.columns if col != "lnl"][0]
    x = df[data_label].values

    # sort df by x
    df = df.sort_values(by=data_label)

    # plot
    fig, axes = plt.subplots(1, 1, figsize=(4, 4))
    axes.scatter(df[data_label], df["lnl"] - df["lnl"].max(), s=0.25)
    axes.plot(df[data_label], df["lnl"] - df["lnl"].max(), lw=0.1)
    axes.axvline(true_val, color="k", linestyle="--", lw=0.5)
    axes.set_ylabel("lnL-max(lnL)")
    axes.set_xlabel(data_label)
    axes.set_ylim(-2, 0)
    return fig


def main():

    grid_fn = "aSF_grid_data.h5"

    for idx in IDX:
        mock_uni = Universe.from_hdf5(grid_fn, idx)
        mock_population = mock_uni.sample_possible_event_matrix()

        outdir = f"mock_uni_{idx}"
        cache = get_training_lnl_cache(
            outdir=outdir + "/large_cache",
            det_matrix_h5=grid_fn,
            mock_uni=mock_population.universe,
            clean=False,
        )

        df = pd.DataFrame(dict(lnl=cache.dataframe["lnl"].values, aSF=cache.dataframe["aSF"].values))
        fig = plot_1d(df, mock_population.aSF)
        fig.savefig(f"{outdir}/aSF.png", dpi=300, bbox_inches="tight")
        mock_population.plot(fname=f"{outdir}/mock_uni.png")
        vertical_concat([f"{outdir}/mock_uni.png", f"{outdir}/aSF.png"], f"{outdir}/mock_uni_and_lnl.png", rm_orig=True)


def save_gif():
    fnames = []
    aSF_val = []
    grid_fn = "aSF_grid_data.h5"
    for i in IDX:
        fnames.append(f'mock_uni_{i}/mock_uni_and_lnl.png')
        mock_uni = Universe.from_hdf5(grid_fn, i)
        aSF_val.append(mock_uni.SF[0])

    df = pd.DataFrame(dict(fnames=fnames, aSF=aSF_val))
    df = df.sort_values(by='aSF')
    make_gif(image_fnames=df.fnames, fname='mock_uni_and_lnl.gif', duration=200, loop=False)


if __name__ == "__main__":
    main()
    save_gif()


