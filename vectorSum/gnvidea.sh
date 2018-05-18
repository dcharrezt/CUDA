#!/bin/bash
#PBS -N vect_sum_256
#PBS -l nodes=1:ppn=4:gpus=2

glxinfo

lshw -C display

lspci | grep ' VGA ' | cut -d" " -f 1 | xargs -i lspci -v -s {}


