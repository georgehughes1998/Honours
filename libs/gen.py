import torch


# Function to predict the next word using a given model
# Returns either the word or a probability distribution of the vocabulary
def predict_next_word(model, dataset, prompt, return_ix=False, return_distribution=False):
    # Run the model
    output = model(dataset.get_tensor_from_string(prompt))

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


def greedy_search(model, dataset, prompt, number_to_generate, return_as_string=True):
    result_list = prompt

    with torch.no_grad():
        for w in range(number_to_generate):
            next_word = predict_next_word(model, dataset, result_list)
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


def random_sample():
    raise NotImplementedError()


def beam_search():
    raise NotImplementedError()