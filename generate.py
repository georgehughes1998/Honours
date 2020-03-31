import torch
import sys

from libs.data_manager_tag import DatasetManagerTag
from model import RNN, load_state_dict, MultiTaskRNN
from libs.data_manager import DatasetManager
from libs.gen import random_sample, greedy_search
from abc2midi import write_abc_file, generate_midi_file, generate_ext_file
from project_constants import *

USE_MULTI_TASK = True

NUM_SAMPLE_TO_GENERATE = 20
MAX_GEN_LENGTH = 200

OUTPUT_PATH = "output/"
ABC_PATH = OUTPUT_PATH + "abc/"
MIDI_PATH = OUTPUT_PATH + "midi/"
PNG_PATH = OUTPUT_PATH + "png/"


def write_abc(file_name, abc_string, title, do_print=False):
    write_abc_file(ABC_PATH + file_name, abc_string, title)
    generate_midi_file(ABC_PATH + file_name, MIDI_PATH + file_name, do_print=do_print)
    generate_ext_file(MIDI_PATH + file_name, PNG_PATH + file_name, file_extension="png", do_print=do_print)


def remove_symbols(the_string):
    return_string = the_string.replace(start_symbol + " ", '').replace(end_symbol, '')
    remove_list = [s.strip() for s in return_string.splitlines()]

    return_string = ""
    for s in remove_list:
        return_string += s + "\n"

    return_string = return_string[:-1]
    return return_string


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device to use:", device)
print()

if USE_MULTI_TASK:
    dataset_info_path = DATASET_TAG_INFO_PATH
else:
    dataset_info_path = DATASET_INFO_PATH

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

start_symbol = dataset.get_start_symbol()
end_symbol = dataset.get_end_symbol()

start_prompt_string = start_symbol + " "


result = greedy_search(rnn, dataset, start_prompt_string.split(), MAX_GEN_LENGTH)
print("Greedy Search with prompt='{}'\n{}".format(start_prompt_string, result))
print()
write_abc("greedy", result, "greedy")


for i in range(NUM_SAMPLE_TO_GENERATE):
    seed = (i+20)*15
    title = str(seed)

    result = random_sample(rnn, dataset, start_prompt_string.split(), MAX_GEN_LENGTH, seed_value=seed)
    result = remove_symbols(result)

    print("Sample {}: Random Sample with prompt='{}', seed={}\n{}".format(i, start_prompt_string, seed, result))

    try:
        write_abc("sample{}".format(i), result, title, do_print=False)
    except AssertionError:
        print("This ABC string was probably not valid.")
