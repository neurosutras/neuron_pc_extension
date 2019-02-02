#!/bin/bash -l

#SBATCH -J test_pc_submit_limits_20190202
#SBATCH -o test_pc_submit_limits_20190202.%j.o
#SBATCH -e test_pc_submit_limits_20190202.%j.e
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -L SCRATCH
#SBATCH -C haswell
#SBATCH -t 00:30:00

set -x

srun -N 1 -n 32 -c 2 --cpu_bind=cores python test_pc_submit_limits.py --block-size=10000 --task-limit=3000000
