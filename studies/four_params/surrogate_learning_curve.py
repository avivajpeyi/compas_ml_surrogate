import random

from compas_surrogate.surrogate.diagnostics.learning_curve import (
    plot_learning_curve_for_lnl,
)

OUTDIR = "out_learning_curve"
H5 = "det_matrix.h5"
random.seed(1)

if __name__ == "__main__":
    plot_learning_curve_for_lnl(OUTDIR, H5, 0, [100, 300, 500, 700, 1000])
