import sys
import string
from os import path
from random import sample

from libs.data_manager import DatasetManager, ALL_DATA
from libs.data_manager_tag import DatasetManagerTag
from project_constants import DATASET_INFO_PATH, DATASET_TAG_INFO_PATH, DATASET_FILE_PATHS, DATASET_SPLIT

DATASET_INFO_PATH = "../" + DATASET_INFO_PATH
DATASET_TAG_INFO_PATH = "../" + DATASET_TAG_INFO_PATH

# Maybe make MIN 1 since resulting dataset is quite small with 2
MIN_NUMBER_SECTIONS = 2
MAX_NUMBER_SECTIONS = 5


def get_struct(piece, sig_symbol):
    # Remove key and time information
    sig_info = piece[:2]
    piece = piece[2:]
    orig_piece = piece
    piece = iter(piece)

    # Only use pieces which go up to Nth section
    section_letter = iter(string.ascii_uppercase[:MAX_NUMBER_SECTIONS])

    bars = []
    last_bar = []
    last_was_num_sec = False
    for p in piece:
        if p == "|":
            bars += [last_bar]
            last_bar = [p]  # + " "
        elif p == "|:":
            bars += [last_bar]
            last_bar = [p]  # + " "
        elif p == ":|":
            if last_was_num_sec:
                last_was_num_sec = False
                last_bar += [p]  # + " "
            else:
                bars += [last_bar + [p]]
                last_bar = []
        elif p == "|1":
            bars += [last_bar]
            last_bar = [p]  # + " "
            last_was_num_sec = True
        elif p == "|2":
            bars += [last_bar]
            last_bar = [p]  # + " "
        else:
            last_bar += [p]  # + " "
    if not bars[-1][-1] == p:
        bars[-1] += p
    # if not "|" in p:
    #     bars[-1] += p
    # else:
    #     bars[-1] += [p]

    # print("Bars", bars)

    # Anacrusis
    bars = iter(bars)
    new_bars = []
    for b in bars:
        if len(b) < 4 and sum([int(d) for d in b if d.isdigit()]) < 4:
            try:
                new_bars.append(b + next(bars))
            except:
                new_bars.append(b)
        else:
            new_bars.append(b)
    bars = new_bars

    # Sections
    sections = []
    this_section = []
    bars = iter(bars)
    section_bar_count = 0
    for b in bars:
        section_bar_count += 1
        if b[0] == "|1":
            this_section += b
            bar_count = 0
            next_bar = next(bars)
            section_bar_count += 1
            this_section += next_bar
            while next_bar[0] != "|2":
                bar_count += 1
                next_bar = next(bars)
                section_bar_count += 1
                this_section += next_bar
            section_bar_count -= 1
            for i in range(bar_count):
                this_section += next(bars)
            sections.append((this_section, section_bar_count))
            section_bar_count = 0
            this_section = []

        elif b[-1] == ":|" and this_section != []:
            this_section += b
            sections.append((this_section, section_bar_count))
            section_bar_count = 0
            this_section = []

        else:
            this_section += b
    if this_section:
        sections.append((this_section, section_bar_count))

    named_sections = dict()

    result_piece = []
    for s in sections:
        named_sections[next(section_letter)] = s[0]

        result_piece += s[0]

    if orig_piece != result_piece:
        # print(orig_piece)
        # print(result_piece)
        # print()
        raise(Exception("Resulting piece doesn't match original piece"))

    if len(named_sections) < MIN_NUMBER_SECTIONS:
        raise(Exception("Resulting piece has too few sections."))

    named_sections[sig_symbol] = sig_info

    return named_sections


# Create a dataset manager object to store/load/save info about the dataset
dataset = DatasetManager(save_path=DATASET_INFO_PATH,
                         data_file_path=DATASET_FILE_PATHS,
                         clean_func=None)

# Load the dataset from the dataset info file
try:
    dataset.load()
    print("Successfully loaded dataset information from {}.".format(DATASET_INFO_PATH))
except FileNotFoundError:
    print("Failed to load dataset information from {}.".format(DATASET_INFO_PATH))
    sys.exit(1)

# Display some details about the loaded dataset
print("Vocab size:", dataset.vocab_size)
print("Sentence length:", dataset.max_sentence_len)
print("Dataset size:", dataset.dataset_size)
print()

pieces = dataset.get_cleaned_data(ALL_DATA)
length = dataset.get_dataset_size(ALL_DATA)
print("Using {} tunes.".format(length))
print()


dataset_tag = DatasetManagerTag(save_path=DATASET_TAG_INFO_PATH)
sig_symbol = dataset_tag.get_sig_symbol()
# get_struct(pieces[12])
sections_list = []

c = 0
failure_count = 0
for p in pieces[:]:
    try:
        sections_list += [get_struct(p, sig_symbol)]
    except BaseException as e:
        failure_count += 1
        # print(c, p)
        # print(e)
        # print()
    c += 1

print("{} failures. That is {}% of the dataset.".format(failure_count, round(100*failure_count/length,2)))
print()

print("Some samples:")
for sec in sample(sections_list, 3):
    print(sec)
print()

if path.exists(DATASET_TAG_INFO_PATH):
    if input("The dataset already exists. Rewriting may cause issues. Proceed? (yes/no)") == "yes":
        dataset_tag.load_dataset(sections_list, split=DATASET_SPLIT)
        dataset_tag.save()
else:
    dataset_tag.load_dataset(sections_list, split=DATASET_SPLIT)
    dataset_tag.save()

dataset_tag.load()

# Display some details about the tagged dataset
print("Vocab size:", dataset_tag.vocab_size)
print("Tag vocab size:", dataset_tag.tag_vocab_size)
print("Sentence length:", dataset_tag.max_sentence_len)
print("Dataset size:", dataset_tag.dataset_size)
print()

tensors = dataset.get_tensors_data()
print(tensors)