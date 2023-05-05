"""
This script makes 10 upsampled versions of the COMPAS h5 file, each with N+sqrt(N) DCO binaries and different sampling seeds.
"""
from compas_surrogate.preprocessing.sample_compas_dcos import downsample

downsample("./DCO_data.h5", outdir="downsampled_datasets", n_copies=10)
