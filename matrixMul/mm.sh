#!/bin/bash
#PBS -N mat_mul
#PBS -l nodes=1:ppn=4:gpus=1

cd $PBS_O_WORKDIR

/usr/local/cuda/bin/nvcc -o mat_mul.out mm.cu
./mat_mul.out

