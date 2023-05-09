#!/bin/bash
#SBATCH --job-name=gen_matrices
#SBATCH --time=8:00:00
#SBATCH --output=sampled_datasets/log_wide.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=22G
#SBATCH --account=oz101

module load python-scientific/3.10.4-foss-2022a
source /fred/oz980/avajpeyi/envs/milan_venv/bin/activate

python make_det_matricies.py --compas_file_regex "sampled_datasets/*.h5" --outdir sampled_datasets/muz_sigms_matricies -n 1500
