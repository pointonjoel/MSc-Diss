# Preamble

# Imports
# pip install PyPDF2 openai wikipedia mwparserfromhell transformers torch pandas scipy # tiktoken
import PyPDF2  # For parsing PDF documents!
# import PyMuPDF # For parsing PDFs
import ast  # covert embeddings saved as strings back to arrays
import openai  # OpenAI API
from sentence_transformers import SentenceTransformer
import pandas as pd  # for storing text and embeddings data
import numpy as np  # for df manipulations
import tiktoken  # for counting tokens
from scipy import spatial  # for calculating cosine similarities for search
import wikipedia  # For sourcing Wikipedia article text
import re  # for cutting <ref> links out of Wikipedia articles
import mwparserfromhell  # for splitting Wikipedia articles into sections
from copy import deepcopy  # for copying dataframes
import torch  # for BERT's argmax and tensors
import fitz  # For parsing PDFs
from unidecode import unidecode  # For decoding PDF text
import re  # for cleaning text
from transformers import BertForQuestionAnswering, BertTokenizer  # For BERT's tokeniser and model
from transformers import BartTokenizer, BartForConditionalGeneration  # For BART's tokeniser and model
import torch  # For creating neural networks with GPUs
import logging  # For showing messages in the console
from datasets import load_dataset  # For loading the dataset
from lxml import html  # For extracting text from HTML docs
import re  # For removing whitespace and HTML tag data
from tqdm import tqdm  # For the progress bar
from bs4 import BeautifulSoup  # For extracting HTML tags
import gc  # For RAM/memory management
import evaluate  # For model evaluation (rouge score)
import nltk  # For use of 'punkt'
from transformers import AutoTokenizer  # For tokenising the dataset
from sklearn.model_selection import train_test_split  # For getting correct answerable-non_answerable split
from datasets import Dataset, DatasetDict, load_from_disk, concatenate_datasets
from transformers import AutoModelForSeq2SeqLM
from huggingface_hub.hf_api import HfFolder  # For pushing model to HF
from transformers import Seq2SeqTrainingArguments  # For model training
from transformers import TextDataset, DataCollatorForLanguageModeling  # For MLM training
from transformers import GPT2Tokenizer, GPT2LMHeadModel  # For MLM training
from transformers import Trainer, TrainingArguments  # For MLM training
from nltk.tokenize import sent_tokenize  # For model evaluation/rouge score
from transformers import DataCollatorForSeq2Seq  # For model training
from transformers import Seq2SeqTrainer  # For model training
import time  # For batch querying GPT API
from transformers import set_seed  # For loading models from a seed
from transformers import pipeline  # For using models on the HF hub
from transformers import EarlyStoppingCallback  # To prevent overfitting of a model

# Logging and GPU setup
logging.basicConfig(filename='main.log', level=logging.DEBUG)  # , encoding='utf-8'


def log_and_print_message(msg):
    print(msg)
    logging.warning(msg)


# Config
CHATBOT_TOPIC = 'Computer Vision'
OUTPUT_DIR = '/content/drive/MyDrive/Diss/Output'
GPT_EMBEDDING_MODEL = "text-embedding-ada-002"
GENERAL_EMBEDDING_MODEL = SentenceTransformer('all-mpnet-base-v2')
GPT_MODEL = "gpt-3.5-turbo"
T5_MODEL = "google/mt5-small"
BART_MODEL = "facebook/bart-large-xsum"
# BERT_MODEL = "deepset/bert-base-cased-squad2"
# BART_MODEL = 'vblagoje/bart_lfqa'
MLM_HF_REFERENCE = 'psxjp5/mlm'
T5_QA_HF_REFERENCE = 'psxjp5/mt5-small'
T5_QA_GPT_HF_REFERENCE = 'psxjp5/mt5-small_gpt_ans'
BART_QA_GPT_HF_REFERENCE = 'psxjp5/mt5-small_gpt_ans'
# BERT_ENCODING = BertTokenizer.from_pretrained(BERT_MODEL)
GPT_TOKENISER = tiktoken.encoding_for_model(GPT_MODEL)
T5_TOKENISER = AutoTokenizer.from_pretrained(T5_QA_GPT_HF_REFERENCE)
BART_TOKENISER = AutoTokenizer.from_pretrained(BART_QA_GPT_HF_REFERENCE)
# BART_ENCODING = BartTokenizer.from_pretrained(BART_MODEL)
GPT_MAX_SECTION_TOKENS = 1600  # max number of tokens per section
T5_MAX_SECTION_TOKENS = 1024  # max number of tokens per section
BART_MAX_SECTION_TOKENS = 1024  # max number of tokens per section
GPT_QUERY_TOKEN_LIMIT = 4096 - 500  # Allows 500 for the response
BERT_MAX_SECTION_TOKENS = 460  # max tokens per section, allowing
# Need to include a check to ensure that the section length is less than the query length (plus the preamble for GPT)
MIN_LENGTH = 50  # min CHARACTER length for each section
SEED = 9  # For reproducibility of model training
RPM = 60  # The RPM limit for querying GPT
NO_ANS_TOKEN = '[NO_ANS]'  # For training the QA model to detect unanswerable questions
ANSWER_NOT_FOUND_MSG = "I could not find an answer in the text I\'ve been provided, sorry! Please try again."

# Downloading metrics for model evaluation
rouge_score = evaluate.load("rouge")
sacrebleu = evaluate.load("sacrebleu")
meteor = evaluate.load('meteor')
nltk.download('punkt')


def add_special_tokens(raw_tokeniser):
    if raw_tokeniser.pad_token is None:
        raw_tokeniser.add_special_tokens({'pad_token': '[PAD]'})
    raw_tokeniser.add_special_tokens({'additional_special_tokens': [NO_ANS_TOKEN]})
    return raw_tokeniser

# SEARCH THE WHOLE PROJECT FOR 'TOKENISER'



class DocTypeNotFoundError(LookupError):
    """
    Raise this when there's an error with the doctype not being specified
    """
    pass


class ModelNotSupportedError(LookupError):
    """
    Raise this when the chosen model isn't supported with a tokeniser
    """
    pass


WIKI_PAGES = [
    'Computer vision'
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

Lec1QAPairs = [
    ["Q: What is the name of this module?", "A: This module is called Introduction to Image Processing"],
    ["Q: What is this module's code?", "A: The module code is COMP 2032"],
    ['Q: Can you help me with my coursework?', "A: I can't directly help you with your coursework but I can assist "
                                               "with your understanding of any relevant concepts!"],
    ["Q: When and where are the lectures held?", "A: Lectures will be held at F3A04 (Block F3, Level A, Room 04) on "
                                                 "Wednesdays (09:00 to 11:00)"],
    ["Q: When and where are the labs held?", "A: Labs will be held at BB80 (Block B, Level B, Room 80) on Tuesdays ("
                                             "14:00 to 16:00)"],
    ["Q: What sources can I use for wider reading?", "A: Digital Image Processing by R.C. Gonzalez and R.E. Woods. ("
                                                     "2018, Fourth Edition), Fundamentals of Digital Image "
                                                     "Processing: A Practical Approach with Examples in Matlab by "
                                                     "Chris Solomon and Toby Breckon (2010), Hypermedia Image "
                                                     "Processing Reference (HIPR) at the University of Edinburgh"],
    ["Q: Who is the lecturer for this module?", "A: This module is taught by Dr Tissa Chandesa ("
                                                "Tissa.Chandesa@nottingham.edu.my)"],
    ["Q: Who are the teaching assistants for this module?", "A: Mr Mahmood Haithami (hcxmh1@nottingham.edu.my) and Mr "
                                                            "Muhammad Waqas (hcxmw1@nottingham.edu.my)"],
    ["Q: What are the assessments for this module?", "A: Coursework (40% weighting) which comprises of Programming "
                                                     "and 2000 Word Report and a 1-hour Written Examination (60%), "
                                                     "where you must answer all 3 questions"],
    ["Q: What is the deadline for the coursework?", "A: Please check moodle for the coursework deadline(s)."],
    ["Q: What is image analysis?", "A: Image analysis is concerned with making quantitative measurements on images, "
                                   "using image measurements as a proxy for real-world values"],
    ["Q: What are the limitations of the techniques covered in this module?", "A: Not all solutions can be reached "
                                                                              "using the covered techniques."],
    ["Q: What can image analysis be used for?", "A: medicine, science, manufacturing, food, textiles"],
    ["Q: Is image processing the same as computer graphics?", "A: No"],
    ["Q: What are some actions that can be performed on digital images using image processing?",
     "A: Acquire an image, store, manipulate, model, analyze, interpret, and display an image."],
    ["Q: What file types will we use for image processing?", "A: jpeg or png"],
    ["Q: Can image processing be applied to videos?", "A: Yes"],
    ["Q: What's the primary aim of this module?", "A: Introduce students to the fundamentals of digital image "
                                                  "processing theory and practice by gaining practical experience in "
                                                  "writing programs to manipulate digital images. It lays down the "
                                                  "foundation for studying advanced topic in related fields"],
    ["Q: What are pictures represented by?", "A: Pixels"],
    ["Q: What colours are used in the greyscale plane?", "A: It is a range from white to black"],
    ["Q: What three colours are used to construct an image?", "A: Red, green and blue"],
    ["Q: What are the three components of image processing?", "A: Image formation, acquisition, colour representation"],
    ["Q: What is image compression used for?", "A: To efficiently represent image data for storage and communication, "
                                               "to minimise disk space and network bandwidth"],
    ["Q: What is the effect of image compression",
     "A: It reduces the image size but makes the quality poorer and more blurry"],
    ["Q: What is image manipulation used for?", "A: Image manipulation is used to remove noise, sharpen, sharpen and "
                                                "enhance or change the contrast and general appearance of an image."],
    ["Q: What are some examples of image compression formats?", "A: Examples include GIF, JPEG, and PNG."],
    ["Q: What are superpixels?", "A: A grouping of similar pixels used as an intermediate image representation to "
                                 "reduce the number of pixels"],
    ["Q: What technique can be used to find geometric objects?", "A: The Hough Transform"],
    ["Q: What are spatial domain methods?", "A: Methods which operate directly on the image, such as point operations "
                                            "and area operations."],
    ["Q: What is the purpose of image segmentation?",
     "A: Image segmentation is used to extract specific objects or regions from an image."],
    ["Q: What is the frequency domain method in image processing?",
     "A: Frequency domain methods involve computing and modifying the power spectrum of an image."],
    ["Q: What is the importance of geometric operations in image processing?", "A: Geometric operations allow for "
                                                                               "changes in the image's array "
                                                                               "structure, and include the "
                                                                               "manipulation of the orientation, "
                                                                               "rotation, and scaling."],
    ["Q: What is content-based image retrieval?", "A: Content-based image retrieval is a technique for searching "
                                                  "large image databases based on their visual content."],
    ["Q: What is painterly rendering in image processing?", "A: Painterly rendering involves processing an image to "
                                                            "give it the appearance of a painting, based on the work "
                                                            "of a particular artist or movement (e.g. "
                                                            "impressionism)."],
    ["Q: What are interactive tools and compositing in image processing?", "A: Interactive tools and compositing "
                                                                           "involve overlaying and combining multiple "
                                                                           "images into a single output image."],
    ["Q: What are some programming languages commonly used for image processing?",
     "A: Commonly used languages include MATLAB, Python (with libraries like PIL and OpenCV), and Java."],
]

Lec2QAPairs = [
    ["Q: What are the two important processes in digital image formation?", "A: Sampling and quantization."],
    ["Q: What is sampling in digital image formation?",
     "A: Sampling is the process of digitizing the spatial coordinates of an image."],
    ["Q: What is quantization in digital image formation?",
     "A: Quantization is the process of digitizing the light intensity function of an image."],
    ["Q: What is aliasing and what causes it?",
     "A: Aliasing is an artifact that occurs when the sampling rate is insufficient, causing the image to become "
     "unrecognizable. It is caused by undersampling or sampling at a rate below the Nyquist rate."],
    ["Q: How many pixels are in an image?",
     "A: The number of pixels in an image depends on its resolution and size. It can vary from image to image."],
    ["Q: How many samples should you take from an image?",
     "A: Samples must be taken at a rate that is twice the frequency of the highest frequency component to be "
     "reconstructed"],
    ["Q: What is the Nyquist Rate?",
     "A: The minimum sampling rate required to accurately reconstruct an image from its sampled version"],
    ["Q: What is under-sampling?", "A: Sampling at a rate which is too course, i.e. below the Nyquist Rate."],
    ["Q: What causes Aliasing?", "A: Aliasing is called by under-sampling."],
    ["Q: What is unsampling",
     "A: The process of reconstructing an image by interpolating pixel values from the sample values"],
    ["Q: What is super resolution?",
     "A: Super resolution methods involve combining multiple exposures of the same scene to enhance the resolution "
     "and quality of an image."],
    ["Q: What is a Bayer pattern?",
     "A: An array of red, green, and blue color filters arranged in a specific repeating pattern."],
    ["Q: Out of red, green and blue, which colour are our eyes drawn to most?",
     "A: Human eyes are most drawn to the colour green."],
    ["Q: What is color intensity in an image?",
     "A: Color intensity refers to the level of intensity or brightness of a color in an image. It determines the "
     "perceived lightness or darkness of a color."],
    ["Q: How can aliasing be avoided in image sampling?",
     "A: Aliasing can be avoided in image sampling by ensuring that the sampling rate is at least twice the frequency "
     "of the highest frequency component to be reconstructed."],
    ["Q: How does interpolation help in image sampling?",
     "A: Interpolation helps in image sampling by estimating unknown pixel values based on known neighboring pixel "
     "values, filling in the gaps and producing a smoother representation of the image."],
    ["Q: Can grey level resolution be increased in a single image?",
     "A: No, grey level resolution cannot be increased in a single image"],
    ["Q: Why is greyscale used in image processing",
     "A: It makes processing easier, reduces the amount of information in the image, and makes some of the theory "
     "simpler."],
    ["Q: How can you convert between colour and greyscale images?",
     "A: By using a weighted average of the red, green, and blue components, with a higher weighting on green (30% "
     "red, 59% green, and 11% blue)."],
    ["Q: What is HSV color space?",
     "A: HSV color space stands for hue, saturation, and value. It is a color model that represents colors based on "
     "their hue (the dominant wavelength), saturation (purity of color), and value (brightness). It is based on "
     "colour rather than light."],
    ["Q: Is HSV more or less sensitive to illumination changes relative to RGB?", "A: Less sensitive"],
    ["Q: What two components of HSV is human skin most captured by, out of hue, saturation, and value?",
     "A: Human skin is most captured by the hue and saturation components of HSV."],
    ["Q: What is an intensity transform?",
     "A: An intensity transform alters the intensity values of pixels in an image, mapping the original pixel "
     "intensities to new intensity values. It is used to manipulate the brightness, contrast, or overall distribution "
     "of intensity levels in an image. These transformations can be linear or nonlinear, and they are applied to "
     "individual pixels or groups of pixels in the image."],
    ["Q: In an linear intensity transform, what is the interpretation of the a and b parameters?",
     "A: A refers to the gain, and b refers to the bias."],
    ["Q: What does the gain in a linear intensity affect?", "A: The gain affects the contrast"],
    ["Q: What does the bias in a linear intensity affect?", "A: The gain affects the brightness"],
    ["Q: What is negation?",
     "A: Negation refers to the operation of inverting or reversing the pixel intensities of an image. It is also "
     "known as image inversion."],
    ["Q: What is the main benefit of negation?",
     "A: The main benefit of negation is that it makes fine details more visible (e.g. digital mammograms)."],
    ["Q: What is contrast stretching?",
     "A: It is a transformation used to convert pixel intensities from one range to another range."],
    ["Q: What type of transformation is Thresholding?", "A: Thresholding is a form of non-linear transformation."],
    ["Q: What is thresholding?",
     "A: A threshold level where any value above it is accepted, and any value below it is rejected. The division of "
     "acceptance and rejection is used to either preserve intensities or remove intensities to black."],
    ["Q: What is a gamma correction used for?",
     "A: It is used to display an image using a voltage which displays a true representation of the image."],
    ["Q: Do point processes affect an image as a whole, or each pixel independently?",
     "A: Point processes operate on each picel independently"],
    ["Q: Do linear processes affect an image as a whole, or each pixel independently?",
     "A: Linear processes change the appearance of the whole image"],
    ["Q: What are non-linear processes used for?",
     "A: Non-linear transformations are used to differentiate between different object/image regions."]
]
