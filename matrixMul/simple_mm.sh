#!/bin/bash
#PBS -N s_mat_mul
#PBS -l nodes=1:ppn=4:gpus=1

cd $PBS_O_WORKDIR

/usr/local/cuda/bin/nvcc -o s_mat_mul.out simple_mult.cu
./s_mat_mul.out

