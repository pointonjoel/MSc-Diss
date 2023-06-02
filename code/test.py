# Preamble
import PyPDF2 # For parsing PDF documents!
import ast  # covert embeddings saved as strings back to arrays
import openai  # OpenAI API
import pandas as pd  # for storing text and embeddings data
import numpy as np # for df manipulations
# import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
import wikipedia # For sourcing Wikipedia article text
import re  # for cutting <ref> links out of Wikipedia articles
import mwparserfromhell  # for splitting Wikipedia articles into sections
from copy import deepcopy # for copying dataframes
import torch # for BERT's argmax and tensors
from transformers import BertForQuestionAnswering, BertTokenizer # For BERT's tokeniser and model
from transformers import BartTokenizer, BartForConditionalGeneration # For BART's tokeniser and model
import torch # For creating neural networks with GPUs

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch
use_cuda = torch.cuda.is_available()
print(f'Cuda available? {use_cuda}')

if use_cuda:
    print(f'__CUDNN VERSION: {torch.backends.cudnn.version()}')
    print(f'__Number CUDA Devices: {torch.cuda.device_count()}')
    print(f'__CUDA Device Name: {torch.cuda.get_device_name(0)}')
    print(f'__CUDA Device Total Memory: {torch.cuda.get_device_properties(0).total_memory/1e9}')
    # model = NeuralNet(input_size, hidden_size, output_size).to(device)