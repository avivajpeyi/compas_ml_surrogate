import os

from compas_surrogate.data_generation.detection_matrix_generator import (
    generate_set_of_matricies,
)

PATH1 = "/home/avaj040/Documents/projects/compas_ml_surrogate/tests/test_data/Z_all/COMPAS_Output.h5"
PATH2 = "/fred/oz980/avajpeyi/projects/compas_dev/data/Z_all/COMPAS_Output.h5"

if os.path.exists(PATH1):
    compas_h5_path = PATH1
else:
    compas_h5_path = PATH2

generate_set_of_matricies(
    compas_h5_path=compas_h5_path,
    n=10000,
    save_images=False,
    outdir="grid",
    parameters=["muz", "sigma0"],
    save_h5_fname="grid_data.h5",
    # custom_ranges={"muz": (-0.5, -0.45), "sigma0": (0.12, 0.26)},
    grid_parameterspace=True,
)
