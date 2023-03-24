import random

from compas_surrogate.surrogate.models.utils import plot_learning_curve_for_lnl

OUTDIR = "out_learning_curve"
H5 = "det_matrix.h5"
random.seed(1)

if __name__ == "__main__":
    plot_learning_curve_for_lnl(OUTDIR, H5, 0, n_pts=1)
