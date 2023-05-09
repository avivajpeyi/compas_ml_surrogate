#!/bin/bash
#SBATCH --job-name=gen_matrices
#SBATCH --time=20:00:00
#SBATCH --output=sampled_datasets/log_wide.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=22G
#SBATCH --account=oz101

module load git/2.18.0 git-lfs/2.4.0 gcc/9.2.0 openmpi/4.0.2 numpy/1.19.2-python-3.8.5 mpi4py/3.0.3-python-3.8.5
module unload zlib
module load pandas/1.2.2-python-3.8.5
source /fred/oz980/avajpeyi/envs/compas_venv/bin/activate

python make_det_matricies.py --compas_file_regex "sampled_datasets/*.h5" --outdir sampled_datasets/muz_sigms_matricies -n 1500
