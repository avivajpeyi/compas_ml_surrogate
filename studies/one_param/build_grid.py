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
    n=1000,
    save_images=False,
    outdir="aSF_grid",
    parameters=[
        "aSF",
    ],
    save_h5_fname="aSF_grid_data.h5",
    grid_parameterspace=True,
)
