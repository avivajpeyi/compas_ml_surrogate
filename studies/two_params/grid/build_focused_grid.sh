#!/bin/bash
#SBATCH --job-name=focused_grid
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=5:00:00
#SBATCH --output=focused_grid.log
#SBATCH --mem=20G

module load git/2.18.0 git-lfs/2.4.0 gcc/9.2.0 openmpi/4.0.2 numpy/1.19.2-python-3.8.5 mpi4py/3.0.3-python-3.8.5
module unload zlib
module load pandas/1.2.2-python-3.8.5
source /fred/oz980/avajpeyi/envs/compas_venv/bin/activate

python build_focused_grid.py
