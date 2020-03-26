import torch

TRAINING_DATA = "TRAINING"
VALIDATION_DATA = "VALIDATION"
TESTING_DATA = "TESTING"
ALL_DATA = "ALL"

_DATA_PARTITIONS = [TRAINING_DATA, VALIDATION_DATA, TESTING_DATA]


# Turn a piece with sections into a normal piece
def compile_piece(piece):
    result_piece = []
    result_tags = []
    for section in piece:
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

    # def get_cleaned_data(self, partition=TRAINING_DATA):
    #     if partition == ALL_DATA:
    #         return self._partitioned_data[TRAINING_DATA] + \
    #                self._partitioned_data[VALIDATION_DATA] + \
    #                self._partitioned_data[TESTING_DATA]
    #     else:
    #         return self._partitioned_data[partition]

    def get_tensors_data(self, partition=TRAINING_DATA):
        # TODO: Amend
        return [self.get_tensor_from_string(s) for s in self._padded_data[partition]]

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

        self.file_paths = obj_dictionary['file_paths']

        self.vocab = obj_dictionary['vocab']
        self.vocab_size = obj_dictionary['vocab_size']
        self.ix_to_vocab = obj_dictionary['ix_to_vocab']
        self.vocab_to_ix = obj_dictionary['vocab_to_ix']

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

        obj_dictionary['file_paths'] = self.file_paths

        obj_dictionary['vocab'] = self.vocab
        obj_dictionary['vocab_size'] = self.vocab_size
        obj_dictionary['ix_to_vocab'] = self.ix_to_vocab
        obj_dictionary['vocab_to_ix'] = self.vocab_to_ix

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
        # TODO: Test

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
        tag_vocab = {self._pad_symbol, self._no_tag_symbol}

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
        # TODO: Fix
        for d in _DATA_PARTITIONS:

            test_piece = self._partitioned_data[d][0]
            test_piece_compile = compile_piece(test_piece)

            print(test_piece)
            print(test_piece_compile)

            self._padded_data[d] = [[self._start_symbol] + s + [self._end_symbol] for s in self._partitioned_data[d]]

            self.max_sentence_len = max([len(s) for s in self._padded_data[d]] + [self.max_sentence_len])

        for d in _DATA_PARTITIONS:
            self._padded_data[d] = [s + [self._pad_symbol] * (self.max_sentence_len - len(s))
                                    for s in self._padded_data[d]]
