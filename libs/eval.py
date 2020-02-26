import torch
import numpy as np

def calculate_perplexity(model, dataset_lines, pad_ix, device=None):
    # TODO: Confirm this implementation is correct
    num_words = 0
    log_likelihood = 0

    model.eval()
    with torch.no_grad():

        for l in dataset_lines:

            cutoff = [i.item() for i in l].index(pad_ix) - 1
            num_words += cutoff + 1
            line = l.to(device)[:cutoff]

            res = model(line[:-1])

            probs = [res[n, 0, line[n+1].item()].item() for n in range(len(line)-1)]
            log_likelihood = log_likelihood + np.sum(probs)

        # perp = 2**(-(log_likelihood / num_words))
        perp = np.exp(-(log_likelihood / num_words))
        return perp


            # for n in line:
            #     v = n.item()
            #
            #     print(res[v])

            # probs = [res[n.item()] for n in line]
            # print(probs)
            # print(np.exp(probs))
            # print(np.product(probs))

