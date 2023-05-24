#!/bin/bash
#SBATCH -p cs -A cs -q cspg
#SBATCH -c4 --mem=4g
#SBATCH --gres gpu:1

python test.py
