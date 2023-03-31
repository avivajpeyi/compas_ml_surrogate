from compas_surrogate.data_generation.detection_matrix_generator import (
    generate_set_of_matricies,
)

generate_set_of_matricies(
    compas_h5_path="/fred/oz980/avajpeyi/projects/compas_dev/data/Z_all/COMPAS_Output.h5",
    n=10000,
    save_images=False,
    outdir="out_muz_sigma0",
    parameters=["muz", "sigma0"],
    save_h5_fname="focused_data.h5",
    custom_ranges={"muz": (-0.5, -0.45), "sigma0": (0.12, 0.26)},
)
