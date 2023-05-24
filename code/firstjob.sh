#!/bin/bash
#SBATCH -p cs -A cs -q cspg
#SBATCH -c1 –mem=512m
seq 1 5
hostname