import torch
import sys

from model import RNN, save_state_dict, load_state_dict
from libs.data_manager import DatasetManager
from libs.gen import random_sample, greedy_search
from abc2midi import write_abc_file, generate_midi_file, generate_ext_file
from project_constants import *


def write_abc(file_name, abc_string):
    write_abc_file(file_name, abc_string)
    generate_midi_file(file_name, file_name, do_print=False)
    generate_ext_file(file_name, file_name, file_extension="png", do_print=False)


def remove_symbols(the_string):
    the_string = the_string.split(end_symbol)[0]
    return the_string.replace(start_symbol, '')


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

start_symbol = dataset.get_start_symbol()
end_symbol = dataset.get_end_symbol()

start_prompt_string = start_symbol + " "
length = 200

result = greedy_search(rnn, dataset, start_prompt_string.split(), length)
print("Greedy Search with prompt='{}'\n{}".format(start_prompt_string, result))
print()
write_abc("output/greedy", remove_symbols(result))

NUM_SAMPLE_TO_GENERATE = 4
for i in range(NUM_SAMPLE_TO_GENERATE):
    result = random_sample(rnn, dataset, start_prompt_string.split(), length, seed_value=i)

    print("Random Sample with prompt='{}', seed={}\n{}".format(start_prompt_string, i, result))
    print()

    write_abc("output/sample{}".format(i), remove_symbols(result))



# def model_get_probabilities(model, prompt="george"):
#     output = model(dataset.get_tensor_from_string(prompt))
#     probabilities = torch.exp(output[-1].view(-1))
#
#     probabilities = {dataset.ix_to_vocab[c]: probabilities[c].item() for c in range(dataset.vocab_size)}
#     return probabilities
#
#
# def model_generate_tree(model, prompt="george", length=5, prob_cutoff_number=2):
#     if not isinstance(prompt, list):
#         prompt = prompt.split()
#     probabilities = model_get_probabilities(model, prompt=prompt)
#
#     # Get the second largest probability
#     probs = [probabilities[c] for c in probabilities]
#     probs.sort()
#
#     new_probabilities = dict()
#     for i in probs[-prob_cutoff_number-1:]:
#         for c in probabilities:
#             if probabilities[c] >= i:
#                 new_probabilities[c] = probabilities.pop(c)
#                 break
#
#     probabilities = new_probabilities
#
#     if length > 1:
#         leaves = {}
#         for c in probabilities:
#             next_probabilities = model_generate_tree(model, prompt=prompt+[c], length=length - 1)
#             node = (probabilities[c], next_probabilities)
#             leaves[c] = node
#         output = leaves
#     else:
#         for c in probabilities:
#             probabilities[c] = (probabilities[c], {})
#         output = probabilities
#
#     return output
#
#
# # Function to return a generated string from the model
# def generate_text_from_model(model, prompt="george", length=15, max_prompt_length=15):
#     p = 0
#     model.eval()
#     prompt = prompt.split()
#     for i in range(length):
#         with torch.no_grad():
#             input_tensor = dataset.get_tensor_from_string(prompt[p:p+max_prompt_length])
#             output = model(input_tensor)
#             # output = output.permute(0, 2, 1)
#             prompt += [dataset.ix_to_vocab[torch.argmax(output, dim=2)[-1].item()]]
#             p += 1
#     model.train()
#
#     text_prompt = ""
#     for w in prompt:
#         text_prompt += w + ' '
#
#     return text_prompt
#
#
# # Test initial model by generating some strings
# for i in range(1):
#     print(generate_text_from_model(rnn, prompt="george hughes", length=40))
#     print(generate_text_from_model(rnn, prompt="zoe hughes", length=40))
#     print(generate_text_from_model(rnn, prompt="callum davies", length=40))
# print()

# print(generate_text_from_model(rnn, prompt="<S>", length=40))
# print()


# def get_tree_string(tree, substring, probability=1):
#     substrings = []
#
#     if len(tree) == 0:
#         return_value = [(substring, probability)]
#     else:
#         for c in tree:
#             substrings += get_tree_string(tree[c][1], substring + ' ' + c, probability*tree[c][0])
#
#         return_value = substrings
#     return return_value
#
#
# def print_tree(tree, substring, do_sort=True):
#     values = get_tree_string(tree, substring)
#
#     if do_sort:
#         values.sort(key=lambda x: x[1])
#
#     for s,p in values:
#         print(s, "(probability={})".format(p))

# rnn.eval()
# prompt = "<S>"
# tree = model_generate_tree(rnn, prompt , length=10, prob_cutoff_number=10)
# print_tree(tree, substring=prompt, do_sort=True)
