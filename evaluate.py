import torch

import sys

from model import RNN
from libs.data_manager import DatasetManager
from project_constants import *

dataset = DatasetManager(save_path=DATASET_INFO_PATH)

try:
    dataset.load()
    print("Successfully loaded dataset information from {}.".format(DATASET_INFO_PATH))
except FileNotFoundError:
    print("Failed to load dataset information from {}.".format(DATASET_INFO_PATH))
    sys.exit(1)


def model_get_probabilities(model, prompt="george"):
    output = model(dataset.get_tensor_from_string(prompt))
    probabilities = torch.exp(output[-1].view(-1))

    probabilities = {dataset.ix_to_vocab[c]: probabilities[c].item() for c in range(dataset.vocab_size)}
    return probabilities


def model_generate_tree(model, prompt="george", length=5, prob_cutoff_number=2):
    if not isinstance(prompt, list):
        prompt = prompt.split()
    probabilities = model_get_probabilities(model, prompt=prompt)

    # Get the second largest probability
    probs = [probabilities[c] for c in probabilities]
    probs.sort()

    new_probabilities = dict()
    for i in probs[-prob_cutoff_number-1:]:
        for c in probabilities:
            if probabilities[c] >= i:
                new_probabilities[c] = probabilities.pop(c)
                break

    probabilities = new_probabilities

    if length > 1:
        leaves = {}
        for c in probabilities:
            next_probabilities = model_generate_tree(model, prompt=prompt+[c], length=length - 1)
            node = (probabilities[c], next_probabilities)
            leaves[c] = node
        output = leaves
    else:
        for c in probabilities:
            probabilities[c] = (probabilities[c], {})
        output = probabilities

    return output


# Function to return a generated string from the model
def generate_text_from_model(model, prompt="george", length=15, max_prompt_length=15):
    p = 0
    model.eval()
    prompt = prompt.split()
    for i in range(length):
        with torch.no_grad():
            input_tensor = dataset.get_tensor_from_string(prompt[p:p+max_prompt_length])
            output = model(input_tensor)
            # output = output.permute(0, 2, 1)
            prompt += [dataset.ix_to_vocab[torch.argmax(output, dim=2)[-1].item()]]
            p += 1
    model.train()

    textPrompt = ""
    for w in prompt:
        textPrompt += w + ' '

    return textPrompt


rnn = RNN(dataset.vocab_size)

# Load state dict
try:
    rnn.load_state_dict(torch.load(STATE_DICT_PATH, map_location=torch.device('cpu')))
    print("Successfully loaded model state from {}.".format(STATE_DICT_PATH))
except FileNotFoundError:
    print("Failed to load model state.")
print()


# # Test initial model by generating some strings
# for i in range(1):
#     print(generate_text_from_model(rnn, prompt="george hughes", length=40))
#     print(generate_text_from_model(rnn, prompt="zoe hughes", length=40))
#     print(generate_text_from_model(rnn, prompt="callum davies", length=40))
# print()

print(generate_text_from_model(rnn, prompt="<S>", length=40))
print()


def get_tree_string(tree, substring, probability=1):
    substrings = []

    if len(tree) == 0:
        return_value = [(substring, probability)]
    else:
        for c in tree:
            substrings += get_tree_string(tree[c][1], substring + ' ' + c, probability*tree[c][0])

        return_value = substrings
    return return_value


def print_tree(tree, substring, do_sort=True):
    values = get_tree_string(tree, substring)

    if do_sort:
        values.sort(key=lambda x: x[1])

    for s,p in values:
        print(s, "(probability={})".format(p))

rnn.eval()
prompt = "<S>"
tree = model_generate_tree(rnn, prompt , length=10, prob_cutoff_number=10)
print_tree(tree, substring=prompt, do_sort=True)