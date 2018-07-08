#!/bin/bash
#PBS -N tiled_mat_mul
#PBS -l nodes=1:ppn=4:gpus=1

cd $PBS_O_WORKDIR

/usr/local/cuda/bin/nvcc -o tiled_mat_mul.out tile_mult.cu
./tiled_mat_mul.out

