import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from numpy import mean
from math import exp
import sys, re, string

from model import RNN, save_state_dict, load_state_dict
from libs.data_manager import DatasetManager
from libs.gen import greedy_search
from project_constants import *


class LearningRate:
    def __init__(self, initial_lr, epoch=0):
        self.initial_lr = initial_lr
        self.epoch = epoch
        self.counter = 0

    def model_was_saved(self):
        self.counter += 1

    def get_learning_rate(self):
        learning_rate = self.initial_lr * exp(-(self.epoch - (self.epoch-self.counter))/4)
        return learning_rate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device to use:", device)
print()

ALLOWED_CHARS = string.ascii_letters + string.digits + string.punctuation + " "

DATASET_FILE_PATHS = ["data/allabcwrepeats_parsed.txt"]

# Starting learning rate (should be fairly high)
LEARNING_RATE = 5

BATCH_SIZE = 32

SAVE_LOSS_MIN = 500

LOSS_PRECISION = 5
TRAINING_PROMPTS = ["<S>"]
TRAINING_PROMPT_LENGTH = 10

PRINT_INTERVAL = 1
GEN_TEXT_INTERVAL = 20


# Function to run on the dataset lines to "clean" them
def clean_func(dataset_lines):
    # Remove lines of length 0
    dataset_lines = [s for s in dataset_lines if len(s) > 0]
    # Remove lines which store key, meter or title information
    new_dataset_lines = []
    for si in range(len(dataset_lines)-2):
        if dataset_lines[si][0] == "T":
            new_sentence = dataset_lines[si+1] + "\n " + dataset_lines[si+2] + "\n "
            new_sentence += dataset_lines[si+3]
            new_dataset_lines += [new_sentence]

    dataset_lines = new_dataset_lines

    return dataset_lines


# Create a dataset manager object to store/load/save info about the dataset
dataset = DatasetManager(save_path=DATASET_INFO_PATH,
                         data_file_path=DATASET_FILE_PATHS,
                         clean_func=clean_func)

# Load the dataset from the dataset info file
try:
    dataset.load()
    print("Successfully loaded dataset information from {}.".format(DATASET_INFO_PATH))
# Load data and process it from the raw data file
except FileNotFoundError:
    dataset.load_dataset(split=(0.88, 0.1, 0.02))
    print("Loaded and processed dataset.")

    # Save to avoid repeating processing
    dataset.save()
    print("Saved data set information to {}.".format(DATASET_INFO_PATH))

# Display some details about the loaded dataset
print("Vocab size:", dataset.vocab_size)
print("Sentence length:", dataset.max_sentence_len)
print("Dataset size:", dataset.dataset_size)
print()

# Turn the dataset into batches
dataset_tensors_pairs = [[s[:-1], s[1:]] for s in dataset.get_tensors_data()]
loader = DataLoader(dataset_tensors_pairs, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
num_batches = dataset.get_dataset_size() // BATCH_SIZE

# Create an instance of the model with given hyperparameters
rnn = RNN(dataset.vocab_size,
          hidden_size=MODEL_HIDDEN_SIZE,
          embedding_size=MODEL_EMBEDDING_SIZE,
          embeddings_dropout=MODEL_EMBEDDINGS_DROPOUT,
          lstm_dropout=MODEL_LSTM_DROPOUT,
          num_decode_layers=MODEL_NUM_HIDDEN_LAYERS)
rnn.to(device)

# Load state dict
try:
    state_dict, epoch, batch, best_loss = load_state_dict(device)
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
lr = LearningRate(LEARNING_RATE, epoch=epoch)
criterion = nn.NLLLoss(ignore_index=dataset.get_pad_ix())
optimiser = optim.SGD(rnn.parameters(), lr=lr.get_learning_rate())
print("Using learning rate {}.".format(lr.get_learning_rate()))
print()

# Array for tracking average loss of an epoch
loss_arr = []


# Main training loop
while True:

    for input, target in loader:

        optimiser.zero_grad()

        input = input.permute(1, 0).to(device)
        target = target.permute(1, 0).to(device)

        output = rnn(input, batch_size=BATCH_SIZE)
        output = output.permute(0, 2, 1)

        # print(input.shape)
        # print(target.shape)
        # print(output.shape)

        loss = criterion(output, target)

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
            save_state_dict(rnn.state_dict(), epoch, batch, avg_loss)

            lr.model_was_saved()

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
            epoch += 1
            batch = 1

            # Adjust the learning rate every epoch
            learning_rate = lr.get_learning_rate()
            optimiser = optim.SGD(rnn.parameters(), lr=learning_rate)
            sys.stdout.write("\r" + "New learning rate: {}.\n".format(learning_rate))
            sys.stdout.flush()
