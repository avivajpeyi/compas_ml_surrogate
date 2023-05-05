#!/bin/bash
#SBATCH --job-name=sample_datasets
#SBATCH --time=2:00:00
#SBATCH --output=sampled_datasets/log.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=10G
#SBATCH --account=oz101

module load git/2.18.0 git-lfs/2.4.0 gcc/9.2.0 openmpi/4.0.2 numpy/1.19.2-python-3.8.5 mpi4py/3.0.3-python-3.8.5
module unload zlib
module load pandas/1.2.2-python-3.8.5
source /fred/oz980/avajpeyi/envs/compas_venv/bin/activate

python generate_uncertainty_datasets.py --in_compas_h5 "/fred/oz101/alej_runs/Z_all/COMPAS_Output.h5" --outdir "sampled_datasets" --n_copies 10
