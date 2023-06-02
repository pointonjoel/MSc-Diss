#!/bin/bash
#SBATCH -p cs -A cs -q cspg
#SBATCH -c4 --mem=4g
#SBATCH --gres gpu:1

module load nvidia/cuda-11.0
module load nvidia/cudnn-v8.0.180-forcuda11.0
pip install PyPDF2 openai wikipedia mwparserfromhell transformers torch pandas scipy
pip install ./assets/setuptools-67.8.0.tar # to install tiktoken
pip install ./assets/setuptools-rust-1.6.0.tar # to install tiktoken
pip install ./assets/tiktoken-0.4.0.tar.gz # as tiktoken is not available via pip install tiktoken
python test.py
