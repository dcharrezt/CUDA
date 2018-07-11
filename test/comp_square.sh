#!/bin/bash
#PBS -N pbs_square
#PBS -l nodes=1:ppn=4:gpus=1

cd $PBS_O_WORKDIR

/usr/local/cuda/bin/nvcc -o square.out square.cu
./square.out
