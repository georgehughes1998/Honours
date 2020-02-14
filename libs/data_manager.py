import torch


class DatasetManager:
    def __init__(self, save_path=None, data_file_path=None, clean_func=None):

        # Store file paths as a list
        if isinstance(data_file_path, list):
            self.file_paths = data_file_path
        elif data_file_path:
            self.file_paths = [data_file_path]
        else:
            self.file_paths = None

        # Info about dataset vocab
        self.vocab = None
        self.vocab_size = None
        self.ix_to_vocab = None
        self.vocab_to_ix = None

        # Info about dataset shape
        self.max_sentence_len = None
        self.dataset_size = None

        # Function used to process the data read from the file
        if not clean_func:
            self._clean_func = lambda x: x
        else:
            self._clean_func = clean_func

        # Stages of processing data
        self._raw_dataset = None
        self._cleaned_data = None
        self._tensors_data = None

        # Path for saving/loading this object to/from
        self._save_path = save_path

        # Special characters
        self._start_symbol = '<S>'
        self._pad_symbol = '<P>'
        self._end_symbol = '</S>'

    def get_cleaned_data(self):
        return self._cleaned_data

    def get_tensors_data(self):
        return [self.get_tensor_from_string(s) for s in self._cleaned_data]
        # return self._tensors_data

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

        #self._raw_dataset = obj_dictionary['raw_dataset']
        self._cleaned_data = obj_dictionary['cleaned_data']
        #self._tensors_data = obj_dictionary['tensors_data']

    # Load and process a dataset
    def load_dataset(self, do_print=False):
        self._load_data_from_file()
        if do_print: print("Loaded data from file(s).")

        self._clean_data()
        if do_print: print("Cleaned data.")

        self._extract_vocab_from_data()
        if do_print: print("Extracted vocab from data.")

        self._pad_data()
        if do_print: print("Padded data.")

        # self._convert_data_to_tensors()
        # if do_print: print("Converted data into tensors.")

    def get_pad_ix(self):
        return self.vocab_to_ix[self._pad_symbol]

    def get_start_symbol(self):
        return self._start_symbol

    def get_end_symbol(self):
        return self._end_symbol

    def _generate_object_dict(self):
        obj_dictionary = dict()

        obj_dictionary['file_paths'] = self.file_paths

        obj_dictionary['vocab'] = self.vocab
        obj_dictionary['vocab_size'] = self.vocab_size
        obj_dictionary['ix_to_vocab'] = self.ix_to_vocab
        obj_dictionary['vocab_to_ix'] = self.vocab_to_ix

        obj_dictionary['max_sentence_len'] = self.max_sentence_len
        obj_dictionary['dataset_size'] = self.dataset_size

        #obj_dictionary['raw_dataset'] = self._raw_dataset
        obj_dictionary['cleaned_data'] = self._cleaned_data
        #obj_dictionary['tensors_data'] = self._tensors_data

        return obj_dictionary

    # Convert a string into a PyTorch Tensor
    def get_tensor_from_string(self, the_string):
        return torch.tensor([self.vocab_to_ix[c] for c in the_string], dtype=torch.long)

    # Load and store lines of a dataset from given a file or files
    def _load_data_from_file(self):
        if not self.file_paths:
            raise Exception("No file paths were given.")

        dataset = []

        for filename in self.file_paths:
            with open(filename, 'r', encoding='utf-8') as file:
                dataset += file.read().splitlines()

        self._raw_dataset = dataset

    # Run a given "clean" function on the dataset
    def _clean_data(self):
        self._cleaned_data = self._clean_func(self._raw_dataset)
        self._cleaned_data = [s.split() for s in self._cleaned_data]
        self.dataset_size = len(self._cleaned_data)

    # Gather information about vocab used in the dataset
    def _extract_vocab_from_data(self):
        vocab = {self._pad_symbol, self._start_symbol, self._end_symbol}

        for s in self._cleaned_data:
            for w in s:
                vocab.add(w)

        # vocab = set(vocab)
        self.vocab = vocab
        self.vocab_size = len(vocab)

        self.ix_to_vocab = dict(enumerate(vocab))
        self.vocab_to_ix = {self.ix_to_vocab[x]: x for x in self.ix_to_vocab}

    # Add padding to lines in the data so each line is the same length and add start/end symbols
    def _pad_data(self):
        self._cleaned_data = [[self._start_symbol] + s + [self._end_symbol] for s in self._cleaned_data]

        self.max_sentence_len = max([len(s) for s in self._cleaned_data])
        self._cleaned_data = [s + [self._pad_symbol] * (self.max_sentence_len - len(s)) for s in self._cleaned_data]

    # # Convert lines in the data to a list of tensors
    # def _convert_data_to_tensors(self):
    #     self._tensors_data = [self.get_tensor_from_string(s) for s in self._cleaned_data]


# # Test the class
# def test_clean_function(dataset):
#     allowed_chars = string.ascii_letters + string.digits + string.punctuation + " "
#     name_list = ("george hughes", "callum davies", "zoe hughes", "fraser macdonald")
#
#     # Filter dates out using regex
#     date_pattern = "[A-Za-z]{3} [0-9]{1,2}, [0-9]{4}, [0-9]{1,2}:[0-9]{1,2} (AM|PM)"
#     dataset = list(filter(lambda x: re.match(date_pattern, x) is None, dataset))
#
#     # Make all letters lowercase
#     dataset = [s.lower() for s in dataset]
#
#     # Filter all non-allowed chars
#     dataset = [''.join(filter(lambda c: c in allowed_chars, s)) for s in dataset]
#
#     # Append what a person said to their name + filter long strings
#     new_dataset = []
#     for si in range(len(dataset)):
#         s = dataset[si]
#         if s in name_list:
#             s1 = dataset[si + 1]
#             # Filter out some strings
#             if 0 < len(s1) < 200:
#                 new_dataset.append(s + ": " + s1)
#     dataset = new_dataset
#
#     return dataset
#
#
# import string, re
#
# path = "../data/"
# files = ["fb_data_callum.txt",
#          "fb_data_zoe.txt",
#          "fb_data_fraser.txt"]
# test_file_paths = [path + f for f in files]
# test_save_path = "dataset_info.pt"
#
# d = DatasetManager(save_path=test_save_path, data_file_path=test_file_paths, clean_func=test_clean_function)
# d.load_dataset(do_print=True)
#
# d.save()
# d.load()
