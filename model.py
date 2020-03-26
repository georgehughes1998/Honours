import torch
import torch.nn as nn

from project_constants import STATE_DICT_PATH
# import torch.nn.functional as F


# Function to load the trained model state from a file
def load_state_dict(device, state_dict_path):
    state_dict_info = torch.load(state_dict_path, map_location=device)
    state_dict = state_dict_info['state_dict']
    epoch = state_dict_info['epoch']
    batch = state_dict_info['batch']
    loss = state_dict_info['loss']

    return state_dict, epoch, batch, loss


# Function to save the trained model state to a file
def save_state_dict(state_dict, state_dict_path, epoch, batch, loss):
    state_dict_info = dict()
    state_dict_info['state_dict'] = state_dict
    state_dict_info['epoch'] = epoch
    state_dict_info['batch'] = batch
    state_dict_info['loss'] = loss

    torch.save(state_dict_info, state_dict_path)


class RNN(nn.Module):
    def __init__(self, vocab_size,
                 hidden_size=64,
                 embedding_size=64,
                 embeddings_dropout=0.3,
                 lstm_dropout=0.5,
                 num_decode_layers=1):

        super(RNN, self).__init__()

        # self.hidden_size = hidden_size
        # self.embedding_size = embedding_size

        self.char_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.embeddings_dropout = nn.Dropout(embeddings_dropout)

        self.lstm = nn.LSTM(embedding_size, hidden_size)
        self.lstm_dropout = nn.Dropout(lstm_dropout)

        # Module list used to have variable number of hidden layers
        self.decode = nn.ModuleList()
        for L in range(num_decode_layers-1):
            layer = nn.Linear(hidden_size, hidden_size)
            self.decode.append(layer)
        layer = nn.Linear(hidden_size, vocab_size)
        self.decode.append(layer)

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, sentence, batch_size=1):

        embeddings = self.char_embeddings(sentence)
        embeddings = embeddings.view(len(sentence), batch_size, -1)
        embeddings = self.embeddings_dropout(embeddings)

        hidden, _ = self.lstm(embeddings)
        hidden = self.lstm_dropout(hidden)

        for L in self.decode:
            hidden = L(hidden)
        output = hidden

        output = self.softmax(output)

        return output


# For dropout values:
# -https://medium.com/@bingobee01/a-review-of-dropout-as-applied-to-rnns-72e79ecd5b7b
# -http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf


# TODO: Test
class MultiTaskRNN(nn.Module):
    def __init__(self, vocab_size,
                 tagset_size,
                 hidden_size=64,
                 embedding_size=64,
                 embeddings_dropout=0.3,
                 lstm_dropout=0.5,
                 num_decode_layers=1):

        super(MultiTaskRNN, self).__init__()

        # self.hidden_size = hidden_size
        # self.embedding_size = embedding_size

        self.char_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.embeddings_dropout = nn.Dropout(embeddings_dropout)

        self.lstm = nn.LSTM(embedding_size, hidden_size)
        self.lstm_dropout = nn.Dropout(lstm_dropout)

        # Module list used to have variable number of hidden layers
        self.decode = nn.ModuleList()
        for L in range(num_decode_layers-1):
            layer = nn.Linear(hidden_size, hidden_size)
            self.decode.append(layer)
        layer = nn.Linear(hidden_size, vocab_size)
        self.decode.append(layer)

        self.decode_tag = nn.ModuleList()
        for L in range(num_decode_layers-1):
            layer = nn.Linear(hidden_size, hidden_size)
            self.decode_tag.append(layer)
        layer = nn.Linear(hidden_size, tagset_size)
        self.decode_tag.append(layer)

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, sentence, batch_size=1):

        embeddings = self.char_embeddings(sentence)
        embeddings = embeddings.view(len(sentence), batch_size, -1)
        embeddings = self.embeddings_dropout(embeddings)

        hidden, _ = self.lstm(embeddings)
        hidden = self.lstm_dropout(hidden)

        hidden_decode = hidden
        hidden_tag_decode = hidden

        for L in self.decode:
            hidden_decode = L(hidden_decode)
        output = hidden_decode

        for L in self.decode_tag:
            hidden_tag_decode = L(hidden_tag_decode)
        output_tag = hidden_tag_decode

        output = self.softmax(output)
        output_tag = self.softmax(output_tag)

        return output, output_tag
