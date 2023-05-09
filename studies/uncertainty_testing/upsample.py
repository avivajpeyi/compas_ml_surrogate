"""
This script makes 10 upsampled versions of the COMPAS h5 file, each with N+sqrt(N) DCO binaries and different sampling seeds.
"""
from compas_surrogate.preprocessing.upsample_compas_dcos import upsample

upsample("./DCO_data.h5", outdir="upsampled_datasets", n_copies=10)
