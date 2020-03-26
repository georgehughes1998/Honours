import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from numpy import mean
import sys, re, string

from libs.data_manager_tag import DatasetManagerTag, TRAINING_DATA
from model import RNN, save_state_dict, load_state_dict, MultiTaskRNN
from libs.gen import greedy_search
from libs.train import LearningRate
from project_constants import *
# from data_taggers.extract_structure import get_struct

# TODO: Implement

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device to use:", device)
print()

DATASET_FILE_PATHS = ["data/allabcwrepeats_parsed.txt"]

# Starting learning rate (should be fairly high)
LEARNING_RATE = 2

BATCH_SIZE = 32

SAVE_LOSS_MIN = 500

LOSS_PRECISION = 5
TRAINING_PROMPTS = ["<S>"]
TRAINING_PROMPT_LENGTH = 10

PRINT_INTERVAL = 1
GEN_TEXT_INTERVAL = 20


dataset = DatasetManagerTag(save_path=DATASET_TAG_INFO_PATH)

# Load the dataset from the dataset info file
try:
    dataset.load()
    print("Successfully loaded dataset information from {}.".format(DATASET_TAG_INFO_PATH))
# Load data and process it from the raw data file
except FileNotFoundError:
    print("Run extract structure.py to generate the dataset.")

# Display some details about the tagged dataset
print("Vocab size:", dataset.vocab_size)
print("Tag vocab size:", dataset.tag_vocab_size)
print("Sentence length:", dataset.max_sentence_len)
print("Dataset size:", dataset.dataset_size)
print()

# Turn the dataset into batches
dataset_tensors_pairs = [[tune[:-1], (tune[1:], tags[:-1])]
                         for (tune, tags) in dataset.get_tensors_data()]
loader = DataLoader(dataset_tensors_pairs, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
num_batches = dataset.get_dataset_size() // BATCH_SIZE


# Create an instance of the model with given hyperparameters
rnn = MultiTaskRNN(dataset.vocab_size,
                   dataset.tag_vocab_size,
                   hidden_size=MODEL_HIDDEN_SIZE_MULTI,
                   embedding_size=MODEL_EMBEDDING_SIZE_MULTI,
                   embeddings_dropout=MODEL_EMBEDDINGS_DROPOUT_MULTI,
                   lstm_dropout=MODEL_LSTM_DROPOUT_MULTI,
                   num_decode_layers=MODEL_NUM_HIDDEN_LAYERS_MULTI)
rnn.to(device)

# Load state dict
try:
    state_dict, epoch, batch, best_loss = load_state_dict(device, state_dict_path=STATE_DICT_PATH_MULTI)
    print(epoch)
    rnn.load_state_dict(state_dict)

    print("Successfully loaded model state from {}.".format(STATE_DICT_PATH))
    print("Picking up at epoch {}, batch {}.".format(epoch, batch))
    print("Best recorded loss was {}.".format(best_loss))
# Initialise counters if model state can't be loaded
except FileNotFoundError:
    epoch, batch, best_loss = 0, 0, 1000
    print("Failed to load model state.")
print()


# Generate and display something from the model
rnn.eval()
print("Generating with current model:")
gen_str = greedy_search(rnn, dataset, dataset.get_start_symbol().split(), 60, device=device)
print(gen_str)
print()
gen_str = greedy_search(rnn, dataset, dataset.get_start_symbol().split(), TRAINING_PROMPT_LENGTH, device=device)
gen_str = gen_str.replace("\n", " ")
rnn.train()


# Training optimiser
lr = LearningRate(LEARNING_RATE, epoch)
criterion = nn.NLLLoss(ignore_index=dataset.get_pad_ix())
optimiser = optim.SGD(rnn.parameters(), lr=lr.get_learning_rate(epoch))
print("Using learning rate {}.".format(lr.get_learning_rate(epoch)))
print()

# Array for tracking average loss of an epoch
loss_arr = []


# Main training loop
while True:

    for input, (target, target_tags) in loader:

        optimiser.zero_grad()

        input = input.permute(1, 0).to(device)

        target = target.permute(1, 0).to(device)
        target_tags = target_tags.permute(1, 0).to(device)

        output, output_tags = rnn(input, batch_size=BATCH_SIZE)

        output = output.permute(0, 2, 1)
        output_tags = output_tags.permute(0, 2, 1)

        # print("Input shape:", input.shape)
        # print("Target shape:", target.shape)
        # print("Output shape:", output.shape)
        # print("Target tags shape:", target_tags.shape)
        # print("Output tags shape:", output_tags.shape)

        loss = criterion(output, target)
        loss += STRUCTURE_TASK_WEIGHT*criterion(output_tags, target_tags)

        loss.backward()
        optimiser.step()

        # ---Display stuff---

        # List of losses to average
        loss_arr += [loss.item()]
        if len(loss_arr) > num_batches:
            loss_arr = loss_arr[num_batches//10:]
        avg_loss = round(mean(loss_arr), LOSS_PRECISION)

        # Generate some text using the model
        if batch % GEN_TEXT_INTERVAL == 0:
            rnn.eval()
            gen_str = greedy_search(rnn, dataset, dataset.get_start_symbol().split(), TRAINING_PROMPT_LENGTH, device=device)
            gen_str = gen_str.replace("\n", " ")
            rnn.train()

        # Save the model
        if avg_loss < best_loss and len(loss_arr) > SAVE_LOSS_MIN:
            best_loss = avg_loss
            save_state_dict(rnn.state_dict(), STATE_DICT_PATH_MULTI, epoch, batch, avg_loss)

            lr.model_was_saved(epoch)

            # if avg_loss < 2:
            sys.stdout.write("\rSaved model at epoch {}, batch {} with loss {}.\n".format(epoch, batch, avg_loss))
            sys.stdout.flush()

        # Display progress
        if batch % PRINT_INTERVAL == 0:
            avg_loss = round(mean(loss_arr), LOSS_PRECISION)
            percentage = 100 * batch // num_batches

            output_template = "E {} {}%. B {}/{}. L {}. G {}"
            output_string = output_template.format(epoch, percentage, batch, num_batches, avg_loss, gen_str)

            sys.stdout.write("\r" + output_string)
            sys.stdout.flush()

        # Count epochs and batches
        batch += 1
        if batch > num_batches:
            # Adjust the learning rate every epoch
            learning_rate = lr.get_learning_rate(epoch)
            optimiser = optim.SGD(rnn.parameters(), lr=learning_rate)
            sys.stdout.write("\r" + "New learning rate: {}.\n".format(learning_rate))
            sys.stdout.flush()

            epoch += 1
            batch = 1
