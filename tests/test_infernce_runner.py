import os.path

from conftest import TEST_DIR

from compas_surrogate.data_generation.detection_matrix_generator import (
    generate_set_of_matricies,
)
from compas_surrogate.inference_runner import run_inference

TEST_DATA_FN = "det_matrix.h5"


def make_test_data(fname=TEST_DATA_FN, compas_h5_path="compas.h5"):
    fpath = os.path.join(TEST_DIR, fname)
    if os.path.exists(fpath):
        return fpath
    generate_set_of_matricies(
        compas_h5_path=compas_h5_path,
        n=50,
        save_images=False,
        outdir=TEST_DIR,
        parameters=["aSF", "dSF"],
        save_h5_fname=TEST_DATA_FN,
    )
    return fpath


def test_run_inference(test_datapath, tmp_path):
    TMP_PATH = "tmp_cache"
    det_matrix_h5 = make_test_data(compas_h5_path=test_datapath)
    run_inference(
        outdir="out",
        n=50,
        cache_outdir=TMP_PATH,
        det_matrix_h5=det_matrix_h5,
        universe_id=0,
        clean=True,
    )
