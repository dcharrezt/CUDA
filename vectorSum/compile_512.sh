#!/bin/bash
#PBS -N vect_sum_512
#PBS -l nodes=1:ppn=4:gpus=2

cd $PBS_O_WORKDIR

/usr/local/cuda/bin/nvcc -o vector_sum_512.out vector_sum.cu
./vector_sum_512.out 1000000
./vector_sum_512.out 10000000
./vector_sum_512.out 100000000
./vector_sum_512.out 1000000000
./vector_sum_512.out 10000000000
./vector_sum_512.out 100000000000
