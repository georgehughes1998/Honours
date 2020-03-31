import torch
import numpy as np

def calculate_perplexity(model, dataset_lines, pad_ix, device=None):
    num_words = 0
    log_likelihood = 0
    tags_likelihood = None
    tags = None
    res_tags = None

    model.eval()
    with torch.no_grad():

        for l in dataset_lines:

            # Multi-task case
            if isinstance(l, tuple):
                tune, tags = l
            else:
                tune = l

            # Discount padding
            try:
                cutoff = [i.item() for i in tune].index(pad_ix) - 1
            except ValueError:
                cutoff = len(tune)

            num_words += cutoff + 1
            line = tune.to(device)[:cutoff]

            # Run the model
            res = model(line[:-1])

            # Multi-task case
            if isinstance(res, tuple):
                res_tune, res_tags = res
            else:
                res_tune = res

            probs = [res_tune[n, 0, line[n+1].item()].item() for n in range(len(line)-1)]
            log_likelihood = log_likelihood + np.sum(probs)

            if (tags is not None) and (res_tags is not None):
                if not tags_likelihood:
                    tags_likelihood = 0

                tags = tags[:cutoff]

                tag_probs = [res_tags[n, 0, tags[n + 1].item()].item() for n in range(len(tags) - 1)]
                tags_likelihood = tags_likelihood + np.sum(tag_probs)

        # perp = 2**(-(log_likelihood / num_words))
        perp = np.exp(-(log_likelihood / num_words))

        if tags_likelihood:
            perp_tags = np.exp(-(tags_likelihood / num_words))
            return_value = (perp, perp_tags)
        else:
            return_value = perp

        return return_value
