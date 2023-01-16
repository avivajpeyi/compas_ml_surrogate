#!/bin/bash
#SBATCH --job-name=make_matricies
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=24GB
#SBATCH --output=log_det_matrix_%j.out
#SBATCH --dependency=singleton

module gcc/9.2.0 openmpi/4.0.2 numpy/1.19.2-python-3.8.5 mpi4py/3.0.3-python-3.8.5
module unload zlib
module load pandas/1.2.2-python-3.8.5
source /fred/oz980/avajpeyi/envs/compas_venv/bin/activate

python make_detection_matricies.py
