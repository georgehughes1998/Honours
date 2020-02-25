import torch
import sys

from model import RNN, load_state_dict
from libs.data_manager import DatasetManager, VALIDATION_DATA
from libs.eval import calculate_perplexity
from project_constants import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device to use:", device)
print()


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
    state_dict, epoch, batch, best_loss = load_state_dict(device)
    rnn.load_state_dict(state_dict)

    print("Successfully loaded model state from {}.".format(STATE_DICT_PATH))
    print("Loaded model at epoch {}, batch {}.".format(epoch, batch))
    print("Best recorded loss was {}.".format(best_loss))
except FileNotFoundError:
    print("Failed to load model state.")
print()


validation_data = dataset.get_tensors_data(partition=VALIDATION_DATA)[:12]
pad_ix = dataset.get_pad_ix()

perplexity = calculate_perplexity(rnn, validation_data, pad_ix)
print(perplexity)
