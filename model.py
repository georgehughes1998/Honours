import torch
import torch.nn as nn

from project_constants import STATE_DICT_PATH
# import torch.nn.functional as F


# Function to load the trained model state from a file
def load_state_dict(device):
    state_dict_info = torch.load(STATE_DICT_PATH, map_location=device)
    state_dict = state_dict_info['state_dict']
    epoch = state_dict_info['epoch']
    batch = state_dict_info['batch']
    loss = state_dict_info['loss']

    return state_dict, epoch, batch, loss


# Function to save the trained model state to a file
def save_state_dict(state_dict, epoch, batch, loss):
    state_dict_info = dict()
    state_dict_info['state_dict'] = state_dict
    state_dict_info['epoch'] = epoch
    state_dict_info['batch'] = batch
    state_dict_info['loss'] = loss

    torch.save(state_dict_info, STATE_DICT_PATH)


class RNN(nn.Module):
    def __init__(self, charset_size, hidden_size=64, embedding_size=64):

        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

        self.char_embeddings = nn.Embedding(charset_size, embedding_size)
        self.embeddings_dropout = nn.Dropout(0.3)

        self.lstm = nn.LSTM(embedding_size, hidden_size)
        self.lstm_dropout = nn.Dropout(0.5)

        self.decode = nn.Linear(hidden_size, charset_size)

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, sentence, batch_size=1):

        embeddings = self.char_embeddings(sentence)
        embeddings = embeddings.view(len(sentence), batch_size, -1)
        embeddings = self.embeddings_dropout(embeddings)

        hidden, _ = self.lstm(embeddings)
        hidden = self.lstm_dropout(hidden)

        output = self.decode(hidden)

        # section_output = self.section_decode(hidden)

        output = self.softmax(output)
#
        # section_output = self.softmax(section_output)
        return output  # , section_output


# For dropout values:
# -https://medium.com/@bingobee01/a-review-of-dropout-as-applied-to-rnns-72e79ecd5b7b
# -http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf
