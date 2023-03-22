"""Plots the o3_h1, o3_l1, o3_v1 sensitivity curves"""

from typing import Dict

import h5py
import matplotlib.pyplot as plt
import numpy as np

H1_SENSITIVITY_FILE = "o3_h1.txt"
L1_SENSITIVITY_FILE = "o3_l1.txt"
V1_SENSITIVITY_FILE = "o3_v1.txt"
COMPAS_SENSITIVITY = "SNR_Grid_IMRPhenomPv2_FD_all_noise.hdf5"


def read_sensitivity_file(fname: str) -> Dict[str, np.ndarray]:
    """Reads the sensitivity file and returns the data"""
    # if fname ends with .txt
    if fname.endswith(".txt"):
        data = np.loadtxt(fname)
        return dict(frequency=data[:, 0], strain=data[:, 1])
    # if fname ends with .hdf5, read the data from the hdf5 file
    elif fname.endswith(".hdf5"):
        with h5py.File(fname, "r") as f:
            data = f["snr_values"]["SimNoisePSDaLIGOMidHighSensitivityP1200087"]
            raise NotImplementedError("Not implemented yet")
    else:
        raise ValueError("File must be a .txt file")


def plot_sensitivity_curves():
    """Plots the sensitivity curves"""
    h1_data = read_sensitivity_file(H1_SENSITIVITY_FILE)
    l1_data = read_sensitivity_file(L1_SENSITIVITY_FILE)
    v1_data = read_sensitivity_file(V1_SENSITIVITY_FILE)
    # compas_data = read_sensitivity_file(COMPAS_SENSITIVITY)

    fig, ax = plt.subplots()
    ax.plot(h1_data["frequency"], h1_data["strain"], label="H1")
    ax.plot(l1_data["frequency"], l1_data["strain"], label="L1")
    ax.plot(v1_data["frequency"], v1_data["strain"], label="V1")
    # ax.plot(compas_data["frequency"], compas_data["strain"], label="COMPAS")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Strain [1/Hz$^{1/2}$]")
    ax.legend(frameon=False)
    plt.show()


if __name__ == "__main__":
    plot_sensitivity_curves()
