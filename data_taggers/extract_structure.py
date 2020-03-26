import sys
import string
from random import sample

from libs.data_manager import DatasetManager, ALL_DATA
from libs.data_manager_tag import DatasetManagerTag
from project_constants import DATASET_INFO_PATH, DATASET_TAG_INFO_PATH, DATASET_FILE_PATHS, DATASET_SPLIT

DATASET_INFO_PATH = "../" + DATASET_INFO_PATH

MAX_NUMBER_SECTIONS = 5


def get_struct(piece):
    # Remove key and time information
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


# get_struct(pieces[12])
sections_list = []

c = 0
failure_count = 0
for p in pieces[:]:
    try:
        sections_list += [get_struct(p)]
    except BaseException as e:
        failure_count += 1
        # print(c, p)
        # print(e)
        # print()
    c += 1

print("{} failures. That is {}% of the dataset.".format(failure_count, 100*failure_count/length))


for sec in sample(sections_list, 3):
    print(sec)

dataset_tag = DatasetManagerTag()
dataset_tag.load_dataset(sections_list, split=DATASET_SPLIT)
