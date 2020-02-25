import torch
import numpy as np

def calculate_perplexity(model, dataset_lines, pad_ix, device=None):
    # TODO: Fix
    num_words = 0
    prod = 1

    model.eval()
    with torch.no_grad():

        for l in dataset_lines:

            cutoff = [i.item() for i in l].index(pad_ix)
            num_words += cutoff
            line = l.to(device)[:cutoff]

            res = model(line[:-1])

            probs = [res[n, 0, line[n+1].item()].item() for n in range(len(line)-1)]
            prod = prod * np.product(probs)

        perp = 2**(-(prod / num_words))
        return perp


            # for n in line:
            #     v = n.item()
            #
            #     print(res[v])

            # probs = [res[n.item()] for n in line]
            # print(probs)
            # print(np.exp(probs))
            # print(np.product(probs))

