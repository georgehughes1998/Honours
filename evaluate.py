import torch
import sys

from model import RNN, load_state_dict, MultiTaskRNN
from libs.data_manager_tag import DatasetManagerTag
from libs.data_manager import DatasetManager, VALIDATION_DATA
from libs.eval import calculate_perplexity
from project_constants import *

USE_MULTI_TASK = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device to use:", device)
print()


if not USE_MULTI_TASK:
    dataset_info_path = DATASET_INFO_PATH
else:
    dataset_info_path = DATASET_TAG_INFO_PATH

if not USE_MULTI_TASK:
    dataset = DatasetManager(save_path=DATASET_INFO_PATH)
else:
    dataset = DatasetManagerTag(save_path=DATASET_TAG_INFO_PATH)

try:
    dataset.load()
    print("Successfully loaded dataset information from {}.".format(dataset_info_path))
except FileNotFoundError:
    print("Failed to load dataset information from {}.".format(dataset_info_path))
    sys.exit(1)


# Create an instance of the model with given hyperparameters
if not USE_MULTI_TASK:
    # Create an instance of the model with given hyperparameters
    rnn = RNN(dataset.vocab_size,
              hidden_size=MODEL_HIDDEN_SIZE,
              embedding_size=MODEL_EMBEDDING_SIZE,
              embeddings_dropout=MODEL_EMBEDDINGS_DROPOUT,
              lstm_dropout=MODEL_LSTM_DROPOUT,
              num_decode_layers=MODEL_NUM_HIDDEN_LAYERS)
else:
    rnn = MultiTaskRNN(dataset.vocab_size,
                       dataset.tag_vocab_size,
                       hidden_size=MODEL_HIDDEN_SIZE_MULTI,
                       embedding_size=MODEL_EMBEDDING_SIZE_MULTI,
                       embeddings_dropout=MODEL_EMBEDDINGS_DROPOUT_MULTI,
                       lstm_dropout=MODEL_LSTM_DROPOUT_MULTI,
                       num_decode_layers=MODEL_NUM_HIDDEN_LAYERS_MULTI)
rnn.eval()

# Load state dict
try:
    if USE_MULTI_TASK:
        state_dict_path = STATE_DICT_PATH_MULTI
    else:
        state_dict_path = STATE_DICT_PATH

    state_dict, epoch, batch, best_loss = load_state_dict(device, state_dict_path)
    rnn.load_state_dict(state_dict)

    print("Successfully loaded model state from {}.".format(state_dict_path))
    print("Loaded model at epoch {}, batch {}.".format(epoch, batch))
    print("Best recorded loss was {}.".format(best_loss))
except FileNotFoundError:
    print("Failed to load model state.")
    sys.exit(1)
print()


validation_data = dataset.get_tensors_data(partition=VALIDATION_DATA)
pad_ix = dataset.get_pad_ix()

print("Calculating perplexity over {} sentences of validation data.".format(len(validation_data)))
perplexity = calculate_perplexity(rnn, validation_data, pad_ix)
if USE_MULTI_TASK:
    perplexity, tag_perplexity = perplexity
print("Calculated perplexity:", perplexity)
if USE_MULTI_TASK:
    print("Calculated perplexity for tags:", tag_perplexity)
