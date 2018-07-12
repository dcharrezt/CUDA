#!/bin/bash
#PBS -N pbs_square
#PBS -l nodes=1:ppn=4

cd $PBS_O_WORKDIR

g++ -o a.out cimg_test.cpp -O2 -L/usr/include/X11 -lm -lpthread -lX11
./a.out
