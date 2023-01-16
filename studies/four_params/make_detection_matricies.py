import os

from compas_surrogate.data_generation import generate_set_of_matricies


def get_compas_output_fname():
    COMPAS_PC = "/home/compas-data/h5out_5M.h5"
    LOCAL_PC = "../../../quasir_compass_blocks/data/COMPAS_Output.h5"
    OZSTAR = (
        "/fred/oz980/avajpeyi/projects/compas_dev/data/Z_all/COMPAS_Output.h5"
    )
    if os.path.exists(COMPAS_PC):
        testfile = COMPAS_PC
    elif os.path.exists(OZSTAR):
        testfile = OZSTAR
    else:
        testfile = LOCAL_PC
    return testfile


def main():
    generate_set_of_matricies(
        n=10000,
        save_images=False,
        compas_h5_path=get_compas_output_fname(),
        outdir="out_aSF_dSF_muz_sigma0",
        parameters=["aSF", "dSF", "muz", "sigma0"],
    )


if __name__ == "__main__":
    main()
