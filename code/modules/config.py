# Preamble

# Imports
# pip install PyPDF2 openai wikipedia mwparserfromhell transformers torch pandas scipy # tiktoken
import PyPDF2 # For parsing PDF documents!
# import PyMuPDF # For parsing PDFs
import ast  # covert embeddings saved as strings back to arrays
import openai  # OpenAI API
from sentence_transformers import SentenceTransformer
import pandas as pd  # for storing text and embeddings data
import numpy as np # for df manipulations
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating vector similarities for search
import wikipedia # For sourcing Wikipedia article text
import re  # for cutting <ref> links out of Wikipedia articles
import mwparserfromhell  # for splitting Wikipedia articles into sections
from copy import deepcopy # for copying dataframes
import torch # for BERT's argmax and tensors
import fitz # For parsing PDFs
from unidecode import unidecode # For decoding PDF text
import re # for cleaning text
from transformers import BertForQuestionAnswering, BertTokenizer # For BERT's tokeniser and model
from transformers import BartTokenizer, BartForConditionalGeneration # For BART's tokeniser and model
import torch # For creating neural networks with GPUs
import logging # For showing messages in the console

# Logging and GPU setup
logging.basicConfig(filename='main.log', level=logging.DEBUG) # , encoding='utf-8'
def log_and_print_message(msg):
        print(msg)
        logging.warning(msg)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
        log_and_print_message(f'Using GPU - details are as follows:')
        log_and_print_message(f'__CUDNN VERSION: {torch.backends.cudnn.version()}')
        log_and_print_message(f'__Number CUDA Devices: {torch.cuda.device_count()}')
        log_and_print_message(f'__CUDA Device Name: {torch.cuda.get_device_name(0)}')
        log_and_print_message(f'__CUDA Device Total Memory: {torch.cuda.get_device_properties(0).total_memory/1e9}')
        # model = NeuralNet(input_size, hidden_size, output_size).to(device)
else:
        log_and_print_message(f'No GPU - available, using a CPU')

# Config
GPT_EMBEDDING_MODEL = "text-embedding-ada-002"
BERT_EMBEDDING_MODEL = 'bert-base-nli-mean-tokens'
GPT_MODEL = "gpt-3.5-turbo"
BERT_MODEL = "deepset/bert-base-cased-squad2"
BART_MODEL = 'vblagoje/bart_lfqa'
GPT_KNOWLEDGE_FILENAME = "CompVisionGPT.csv"
BERT_KNOWLEDGE_FILENAME = "CompVisionBERT.csv"
BERT_ENCODING = BertTokenizer.from_pretrained(BERT_MODEL)
GPT_ENCODING = tiktoken.encoding_for_model(GPT_MODEL)
BART_ENCODING = BartTokenizer.from_pretrained(BART_MODEL)
GPT_MAX_SECTION_TOKENS = 1600 # max number of tokens per section
GPT_QUERY_TOKEN_LIMIT = 4096 - 500 # Allows 500 for the response
BERT_MAX_SECTION_TOKENS = 460 # max tokens per section, allowing
# Need to include a check to ensure that the section length is less than the query length (plus the preamble for GPT)
MIN_LENGTH = 50 # min CHARACTER length for each section
ANSWER_NOT_FOUND_MSG = "I could not find an answer in the text I\'ve been provided, sorry! Please try again."
WIKI_PAGES = [
    'Computer vision',
    'Databases and indexing related concepts',
    'Generic computer vision methods',
    'Geometric and other image features and methods',
    'Geometry and mathematics',
    'Image physics related concepts',
    'Image Processing Architectures & Control Structures',
    'Image transformations and filters',
    'Introductory visual neurophysiology',
    'Introductory visual psychophysics/psychology',
    'Motion and time sequence analysis related concepts',
    'Non-sequential realization methods',
    'Object, world and scene representations',
    'Recognition and registration methods',
    'Scene understanding/image analysis methods',
    'Sensor fusion, registration and planning methods',
    'Sensors and properties',
    'System models, calibration and parameter estimation methods',
    'Visual learning related methods and concepts'
    ]
WIKI_PAGE = "Computer vision"
SECTIONS_TO_IGNORE = [
    "See also",
    "References",
    "External links",
    "Further reading",
    "Footnotes",
    "Bibliography",
    "Sources",
    "Citations",
    "Literature",
    "Footnotes",
    "Notes and references",
    "Photo gallery",
    "Works cited",
    "Photos",
    "Gallery",
    "Notes",
    "References and sources",
    "References and notes",
    ]


class DocTypeNotFoundError(LookupError):
    """
    Raise this when there's an error with the doctype not being specified
    """
    pass