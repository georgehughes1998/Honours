import torch

TRAINING_DATA = "TRAINING"
VALIDATION_DATA = "VALIDATION"
TESTING_DATA = "TESTING"
ALL_DATA = "ALL"

_DATA_PARTITIONS = [TRAINING_DATA, VALIDATION_DATA, TESTING_DATA]


# Turn a piece with sections into a normal piece
def compile_piece(piece, sig_symbol):
    sig = piece[sig_symbol]
    result_piece = sig[:]
    result_tags = [sig_symbol]*2

    for section in piece:
        if section != sig_symbol:
            result_piece += piece[section]
            result_tags += [section] * len(piece[section])
    return result_piece, result_tags


class DatasetManagerTag:
    def __init__(self, save_path=None):

        # Info about dataset vocab
        self.vocab = None
        self.vocab_size = None
        self.tag_vocab = None
        self.tag_vocab_size = None

        self.ix_to_vocab = None
        self.vocab_to_ix = None
        self.ix_to_tag = None
        self.tag_to_ix = None

        # Info about dataset shape
        self.max_sentence_len = 0
        self.dataset_size = dict()

        # Stages of processing data
        self._partitioned_data = dict()
        self._padded_data = dict()

        # Path for saving/loading this object to/from
        self._save_path = save_path

        # Special characters
        self._start_symbol = '<S>'
        self._pad_symbol = '<P>'
        self._end_symbol = '</S>'
        self._no_tag_symbol = '<NT>'
        self._sig_symbol = '<SIG>'

        self._split_char = " "

    def get_tensors_data(self, partition=TRAINING_DATA):
        return [(self.get_tensor_from_string(tune),  self.get_tensor_from_tags(tags))
                for (tune, tags) in self._padded_data[partition]]

    # Get a copy of the strings of data for a partition
    def get_data(self, partition=TRAINING_DATA, include_tags=True):
        if include_tags:
            ret_val = [x for x in self._padded_data[partition]]
        else:
            ret_val = [tune for (tune, tags) in self._padded_data[partition]]
        return ret_val

    def get_dataset_size(self, partition=TRAINING_DATA):
        if partition == ALL_DATA:
            return sum(self.dataset_size.values())
        else:
            return self.dataset_size[partition]

    def save(self):
        obj_dictionary = self._generate_object_dict()
        torch.save(obj_dictionary, self._save_path)

    def load(self):
        obj_dictionary = torch.load(self._save_path)

        self.vocab = obj_dictionary['vocab']
        self.vocab_size = obj_dictionary['vocab_size']
        self.ix_to_vocab = obj_dictionary['ix_to_vocab']
        self.vocab_to_ix = obj_dictionary['vocab_to_ix']

        self.tag_vocab = obj_dictionary['tag_vocab']
        self.tag_vocab_size = obj_dictionary['tag_vocab_size']
        self.ix_to_tag = obj_dictionary['ix_to_tag']
        self.tag_to_ix = obj_dictionary['tag_to_ix']

        self.max_sentence_len = obj_dictionary['max_sentence_len']
        self.dataset_size = obj_dictionary['dataset_size']

        self._partitioned_data = obj_dictionary['partitioned_data']
        self._pad_data()

    # Load and process a dataset
    def load_dataset(self, sections_list, split=(0.8, 0.1, 0.1), do_print=False):
        # Validate split sums to one
        if not sum(split) == 1:
            raise Exception("Split must sum to one.")

        self._load_data_from_sections_list(sections_list, split)
        if do_print: print("Loaded data from sections list.")

        self._extract_vocab_from_data(sections_list)
        if do_print: print("Extracted vocab from data.")

        self._pad_data()
        if do_print: print("Padded data.")

    def get_pad_ix(self):
        return self.vocab_to_ix[self._pad_symbol]

    def get_start_symbol(self):
        return self._start_symbol

    def get_end_symbol(self):
        return self._end_symbol

    def get_sig_symbol(self):
        return self._sig_symbol

    def get_no_tag_symbol(self):
        return self._no_tag_symbol

    def _generate_object_dict(self):
        obj_dictionary = dict()

        obj_dictionary['vocab'] = self.vocab
        obj_dictionary['vocab_size'] = self.vocab_size
        obj_dictionary['ix_to_vocab'] = self.ix_to_vocab
        obj_dictionary['vocab_to_ix'] = self.vocab_to_ix

        obj_dictionary['tag_vocab'] = self.tag_vocab
        obj_dictionary['tag_vocab_size'] = self.tag_vocab_size
        obj_dictionary['ix_to_tag'] = self.ix_to_tag
        obj_dictionary['tag_to_ix'] = self.tag_to_ix

        obj_dictionary['max_sentence_len'] = self.max_sentence_len
        obj_dictionary['dataset_size'] = self.dataset_size

        obj_dictionary['partitioned_data'] = self._partitioned_data

        return obj_dictionary

    # Convert a string into a PyTorch Tensor
    def get_tensor_from_string(self, the_string):
        return torch.tensor([self.vocab_to_ix[c] for c in the_string], dtype=torch.long)

    # Convert a string into a PyTorch Tensor
    def get_tensor_from_tags(self, the_tags):
        return torch.tensor([self.tag_to_ix[c] for c in the_tags], dtype=torch.long)

    def _load_data_from_sections_list(self, sections_list, split):
        total_dataset_size = len(sections_list)

        # Store data in partitions
        start_point = 0
        for i in range(3):
            end_point = int(start_point + (total_dataset_size * split[i]))

            data_partition = sections_list[int(start_point):int(end_point)]
            self._partitioned_data[_DATA_PARTITIONS[i]] = data_partition

            start_point = end_point + 1

            self.dataset_size[_DATA_PARTITIONS[i]] = len(data_partition)

    # Gather information about vocab used in the dataset
    def _extract_vocab_from_data(self, sections_list):
        vocab = {self._pad_symbol, self._start_symbol, self._end_symbol}
        tag_vocab = {self._pad_symbol, self._start_symbol, self._end_symbol, self._no_tag_symbol}

        for piece in sections_list:
            for section in piece:
                for word in piece[section]:
                    vocab.add(word)
                tag_vocab.add(section)

        # vocab = set(vocab)
        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.tag_vocab = tag_vocab
        self.tag_vocab_size = len(tag_vocab)

        self.ix_to_vocab = dict(enumerate(vocab))
        self.vocab_to_ix = {self.ix_to_vocab[x]: x for x in self.ix_to_vocab}

        self.ix_to_tag = dict(enumerate(tag_vocab))
        self.tag_to_ix = {self.ix_to_tag[x]: x for x in self.ix_to_tag}

    # Add padding to lines in the data so each line is the same length and add start/end symbols
    def _pad_data(self):
        self.max_sentence_len = 0
        # For each partition
        for d in _DATA_PARTITIONS:
            self._padded_data[d] = []

            # For each piece
            for piece in self._partitioned_data[d]:
                notes, tags = compile_piece(piece, self._sig_symbol)

                notes = [self._start_symbol] + notes + [self._end_symbol]
                tags = [self._start_symbol] + tags + [self._end_symbol]

                if len(notes) != len(tags):
                    # print(len(notes))
                    # print(len(tags))
                    raise(Exception("Length of notes and tags should be equal. Something has gone horribly wrong."))

                self._padded_data[d].append((notes, tags))

                self.max_sentence_len = max(self.max_sentence_len, len(notes))

        for d in _DATA_PARTITIONS:
            # Comprehension to pad both the piece and tags
            self._padded_data[d] = [
                (piece + [self._pad_symbol] * (self.max_sentence_len - len(piece)),
                 tags + [self._pad_symbol] * (self.max_sentence_len - len(tags)))
                for (piece, tags) in self._padded_data[d]
            ]
