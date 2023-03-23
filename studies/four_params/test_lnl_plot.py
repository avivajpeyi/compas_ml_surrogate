from compas_surrogate.inference_runner import get_training_lnl_cache

cache = get_training_lnl_cache(
    "test0", n_samp=None, det_matrix_h5="det_matrix.h5", universe_id=0
)
cache.plot("test/all_pts_corner.png", show_datapoints=True)

cache = get_training_lnl_cache(
    "test", n_samp=1000, det_matrix_h5="det_matrix.h5", universe_id=0
)
cache.plot("test/corner.png")
cache.plot("test/corner_2.png", show_datapoints=True)
