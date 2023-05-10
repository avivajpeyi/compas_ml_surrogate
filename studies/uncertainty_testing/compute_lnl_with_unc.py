import glob

import click
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from compas_surrogate.data_generation.likelihood_cacher import (
    Universe,
    get_training_lnl_cache,
)


@click.command()
@click.option(
    "--matrix_regex",
    type=str,
    help="Regrex for the matrix files to use",
)
@click.option(
    "--outdir",
    type=str,
    help="Output directory",
)
def make_lnl_table(
    matrix_regex,
    outdir,
):

    matix_paths = glob.glob(matrix_regex)
    print(f"Found {len(matix_paths)} matrices")
    print(f"Loading {matix_paths[1]}")
    mock_uni = Universe.from_hdf5(matix_paths[1], 2)
    mock_population = mock_uni.sample_possible_event_matrix()

    caches = []
    for i in tqdm(range(len(matix_paths)), desc="Loading caches"):

        caches.append(
            get_training_lnl_cache(
                outdir=f"{outdir}/lnl_cache_{i}",
                det_matrix_h5=matix_paths[i],
                mock_uni=mock_population.universe,
                clean=False,
            )
        )
        Universe.from_hdf5(matix_paths[i], 20).plot_detection_rate_matrix(
            fname=f"{outdir}/matrix_{i}.png"
        )

    print("Caches loaded")
    # combine dataframes (rename each lnl column to the matrix id) based on the other columns
    df = caches[0].dataframe.copy()
    # get list of columns to merge on (exclude lnl)
    merge_on = [col for col in df.columns if col != "lnl"]
    df = df.rename(columns={"lnl": f"lnl_{0}"})
    for i, cache in enumerate(caches[1:]):
        curr_df = cache.dataframe.copy().rename(columns={"lnl": f"lnl_{i+1}"})
        df = df.merge(curr_df, on=merge_on)
    print("Dataframes merged")
    print(df)
    true_vals = dict(
        muz=mock_population.muz,
        sigma0=mock_population.sigma0,
        aSF=mock_population.aSF,
        dSF=mock_population.dSF,
    )
    plot_1d_lnl(df, true_vals, "muz")
    if "sigma0" in df.columns:
        plot_1d_lnl(df, true_vals, "sigma0")


def plot_1d_lnl(df, true_vals, parm_name):
    # filter data to only include the true value of other parameters
    for parm, val in true_vals.items():
        if parm != parm_name and parm in df.columns:
            df = df[df[parm] == val]
    df = df.sort_values(by=parm_name)

    # mean of lnl columns
    lnl_mean = df[[col for col in df.columns if col.startswith("lnl")]].mean(axis=1)
    lnl_std = df[[col for col in df.columns if col.startswith("lnl")]].std(axis=1)

    plt.close("all")

    # plot it
    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        gridspec_kw={"height_ratios": [3, 1], "wspace": 0, "hspace": 0},
        sharex=True,
    )

    # make fig sub axes small

    # num of lnl_i columns
    num_lnl = len([col for col in df.columns if col.startswith("lnl")])
    for i in range(num_lnl):
        ax0.scatter(
            df[parm_name], df[f"lnl_{i}"], label=f"matrix {i}", color=f"C{i}", alpha=0.1
        )
        ax0.plot(
            df[parm_name], df[f"lnl_{i}"], label=f"matrix {i}", color=f"C{i}", alpha=0.1
        )
    # plot relative error

    ax1.errorbar(
        df[parm_name],
        [0] * len(df),
        yerr=lnl_std,
        fmt=".",
    )
    ax0.axvline(true_vals[parm_name], label="True muz", color="red")
    ax1.axvline(true_vals[parm_name], label="True muz", color="red")
    ax1.set_xlabel(parm_name)
    ax1.set_ylabel("std(lnl)")
    ax0.set_ylabel("lnl")
    ax0.legend(fontsize=8)
    fig.tight_layout()
    plt.savefig(f"{parm_name}_lnl.png")


# def main():
#     make_lnl_table(
#         "upsampled_datasets/500grid_mu_sig/*.h5",
#         outdir="upsampled_datasets/500grid_mu_sig/lnl_tables",
#     )
#     make_lnl_table(
#         "downsampled_datasets/500grid_mu_sig/*.h5",
#         outdir="downsampled_datasets/500grid_mu_sig/lnl_tables",
#     )
#
#
if __name__ == "__main__":
    make_lnl_table()
