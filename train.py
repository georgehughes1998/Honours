import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from numpy import mean
from random import choice
import sys, re, string

from model import RNN
from libs.data_manager import DatasetManager
from project_constants import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device to use:", device)

ALLOWED_CHARS = string.ascii_letters + string.digits + string.punctuation + " "

DATASET_FILE_PATHS = ["data/allabcwrepeats_parsed.txt"]

LEARNING_RATE = 10

BATCH_SIZE = 32

LOSS_PRECISION = 5
TRAINING_PROMPTS = ["<S>"]
TRAINING_PROMPT_LENGTH = 5

PRINT_INTERVAL = 1
GEN_TEXT_INTERVAL = 5
SAVE_INTERVAL = 100

EPOCHS = 100


# Function to run on the dataset lines to "clean" them
def clean_func(dataset_lines):
    # Remove lines of length 0
    dataset_lines = [s for s in dataset_lines if len(s) > 0]
    # Remove lines which store key, meter or title information
    dataset_lines = [s for s in dataset_lines if not s[0] in ['T', 'M', 'K']]

    return dataset_lines


# Create a dataset manager object to store/load/save info about the dataset
dataset = DatasetManager(save_path=DATASET_INFO_PATH,
                         data_file_path=DATASET_FILE_PATHS,
                         clean_func=clean_func)

try:
    dataset.load()
    print("Successfully loaded dataset information from {}.".format(DATASET_INFO_PATH))

except FileNotFoundError:
    dataset.load_dataset()
    print("Loaded and processed dataset.")

    dataset.save()
    print("Saved data set information to {}.".format(DATASET_INFO_PATH))

print("Vocab size:", dataset.vocab_size)
print("Sentence len:", dataset.max_sentence_len)
# for i in dataset.get_cleaned_data()[:10]:
#     print(i)
print()

# Process the dataset into batches
dataset_tensors_pairs = [[s[:-1], s[1:]] for s in dataset.get_tensors_data()]

loader = DataLoader(dataset_tensors_pairs, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
num_batches = dataset.dataset_size // BATCH_SIZE

# Create an instance of the model
rnn = RNN(dataset.vocab_size)
rnn.to(device)

# Load state dict
try:
    rnn.load_state_dict(torch.load(STATE_DICT_PATH, map_location=device))
    print("Successfully loaded model state from {}.".format(STATE_DICT_PATH))
except FileNotFoundError:
    print("Failed to load model state.")
print()


# Function to return a generated string from the model
def generate_text_from_model(model, prompt="<S>", length=15, max_prompt_length=15):
    p = 0
    model.eval()
    prompt = prompt.split()
    for i in range(length):
        with torch.no_grad():
            input_tensor = dataset.get_tensor_from_string(prompt[p:p+max_prompt_length]).to(device)
            output = model(input_tensor)
            # output = output.permute(0, 2, 1)
            prompt += [dataset.ix_to_vocab[torch.argmax(output, dim=2)[-1].item()]]
            p += 1
    model.train()

    textPrompt = ""
    for w in prompt:
        textPrompt += w + ' '

    return textPrompt


# Test initial model by generating some strings
for s in TRAINING_PROMPTS:
    print(generate_text_from_model(rnn, prompt=s, length=40))
print()

# Train
criterion = nn.NLLLoss(ignore_index=dataset.get_pad_ix())
optimiser = optim.SGD(rnn.parameters(), lr=LEARNING_RATE)

gen_str = generate_text_from_model(rnn)

# TODO: Save best weights


for epoch in range(EPOCHS):

    total_loss = 0
    loss_arr = []
    c = 1

    for input, target in loader:

        optimiser.zero_grad()

        input = input.permute(1,0).to(device)
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
        loss_arr += [loss.item()]  # sum loss

        # Generate some text using the model
        if c % GEN_TEXT_INTERVAL == 0:
            prompt=choice(TRAINING_PROMPTS)
            gen_str = generate_text_from_model(rnn, length=TRAINING_PROMPT_LENGTH, prompt=prompt)

        # Display progress
        if c % PRINT_INTERVAL == 0:
            # avg_loss = round(total_loss / PRINT_INTERVAL, 5)
            avg_loss = round(mean(loss_arr), LOSS_PRECISION)
            percentage = 100 * c // num_batches

            output_template = "Epoch {}: {}% complete. {}/{} processed. Loss={}.\tLast Generated: {}"
            output_string = output_template.format(epoch, percentage, c, num_batches, avg_loss, gen_str)

            sys.stdout.write("\r" + output_string)
            sys.stdout.flush()

            total_loss = 0

        # Save the model
        if c % SAVE_INTERVAL == 0:
            torch.save(rnn.state_dict(), STATE_DICT_PATH)

        c += 1
