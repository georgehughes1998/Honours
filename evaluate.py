import torch
import sys

from model import RNN
from libs.data_manager import DatasetManager
from libs.eval import calculate_perplexity
from project_constants import *


dataset = DatasetManager(save_path=DATASET_INFO_PATH)

try:
    dataset.load()
    print("Successfully loaded dataset information from {}.".format(DATASET_INFO_PATH))
except FileNotFoundError:
    print("Failed to load dataset information from {}.".format(DATASET_INFO_PATH))
    sys.exit(1)


# Create model object
rnn = RNN(dataset.vocab_size)
rnn.eval()

# Load state dict
try:
    rnn.load_state_dict(torch.load(STATE_DICT_PATH, map_location=torch.device('cpu')))
    print("Successfully loaded model state from {}.".format(STATE_DICT_PATH))
except FileNotFoundError:
    print("Failed to load model state.")
print()


perplexity = calculate_perplexity(rnn, dataset)
print(perplexity)