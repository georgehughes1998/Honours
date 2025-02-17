import torch
import re

from string import digits
from numpy.random import choice, seed


time_modifier_pattern = r'[{0}]+|/[{0}]+|[{0}]+/[{0}]+'.format(digits)
time_modifier_pattern = re.compile(time_modifier_pattern)


# Function to predict the next word using a given model
# Returns either the word or a probability distribution of the vocabulary
def predict_next_word(model, dataset, prompt, return_ix=False, return_distribution=False, device=None):
    # Run the model
    input = dataset.get_tensor_from_string(prompt).to(device)
    output = model(input)

    # MultiTask case
    if isinstance(output, tuple):
        output = output[0]

    # Return a probability distribution for the vocabulary
    if return_distribution:
        probabilities = torch.exp(output[-1].view(-1))

        # Return a tensor of probabilities
        if return_ix:
            return_value = probabilities
        # Return a dictionary of {words: probabilities}
        else:
            return_value = {dataset.ix_to_vocab[ix]: probabilities[ix].item()
                            for ix in range(len(probabilities))}

    # Return the next word
    else:
        arg_max = torch.argmax(output, dim=2)[-1].item()

        # Return the most likely word index
        if return_ix:
            return_value = arg_max
        # Return the most likely word string
        else:
            return_value = dataset.ix_to_vocab[arg_max]

    return return_value


def greedy_search(model, dataset, prompt, number_to_generate, return_as_string=True, device=None):
    result_list = prompt

    with torch.no_grad():
        for w in range(number_to_generate):
            next_word = predict_next_word(model, dataset, result_list, device=device)
            result_list += [next_word]

    # Return as a string
    if return_as_string:
        result_string = ""
        for word in result_list:
            result_string += word + ' '
        result_string = result_string[:-1]
        return_value = result_string
    # Return as a list
    else:
        return_value = result_list

    return return_value


def random_sample(model, dataset, prompt, number_to_generate, return_as_string=True, seed_value=0):
    result_list = prompt
    vocab_keys = range(dataset.vocab_size)

    # Use the provided seed value for choosing from the distributions
    seed(seed_value)

    with torch.no_grad():
        for w in range(number_to_generate):
            # Get probability distribution
            probabilities = predict_next_word(model, dataset, result_list, return_distribution=True, return_ix=True)

            # Normalise probabilities so sums to 1
            probabilities = [p.item() for p in probabilities]
            probabilities = [p / sum(probabilities) for p in probabilities]

            # Use the probability distribution to get a random word
            next_word = choice(vocab_keys, p=probabilities)
            next_word = dataset.ix_to_vocab[next_word]

            result_list += [next_word]

            if next_word == dataset.get_end_symbol():
                break

    # Return as a string
    if return_as_string:
        result_string = ""
        for word in result_list:
            if not time_modifier_pattern.match(word):
                result_string += ' '
            result_string += word
        result_string = result_string[1:]

        return_value = result_string
    # Return as a list
    else:
        return_value = result_list

    return return_value


def beam_search():
    raise NotImplementedError()