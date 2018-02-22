#!/bin/bash -l

#SBATCH -J test_pc_extension_20180221
#SBATCH -o test_pc_extension_20180221.%j.o
#SBATCH -e test_pc_extension_20180221.%j.e
#SBATCH -q debug
#SBATCH -N 8
#SBATCH -L SCRATCH
#SBATCH -C haswell
#SBATCH -t 0:30:00

set -x

cd $HOME/neuron_pc_extension
export DATE=$(date +%Y%m%d_%H%M%S)

srun -N 8 -n 256 -c 2 --cpu_bind=cores python test_pc_extension.py
