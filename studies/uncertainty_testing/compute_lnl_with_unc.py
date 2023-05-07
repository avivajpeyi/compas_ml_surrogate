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
    plot_1d_lnl(df, mock_uni.muz, "muz")
    plot_1d_lnl(df, mock_uni.sigma0, "sigma0")


def plot_1d_lnl(df, true_val, parm_name):
    # sort by muz
    df = df.sort_values(by=parm_name)

    # group data by param_name
    df = df.groupby(parm_name).mean()

    # mean of lnl columns
    lnl_mean = df[[col for col in df.columns if col.startswith("lnl")]].mean(axis=1)
    lnl_std = df[[col for col in df.columns if col.startswith("lnl")]].std(axis=1)

    plt.close("all")
    plt.figure()

    # num of lnl_i columns
    num_lnl = len([col for col in df.columns if col.startswith("lnl")])
    vals = df.index.values
    plt.errorbar(vals, lnl_mean, lnl_std, label="LNL mean", color="black", alpha=0.5)
    for i in range(num_lnl):
        plt.plot(vals, df[f"lnl_{i}"], label=f"matrix {i}", color="black", alpha=0.1)
    plt.axvline(true_val, label="True muz", color="red")
    plt.show()


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
