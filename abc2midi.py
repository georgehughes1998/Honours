import os


# Lib-level constants
MUSESCORE_BIN_PATH = "C:/Program Files (x86)/MuseScore 2/bin/MuseScore.exe"
ABC_2_MIDI_PATH = "abc2midi\\abc2midi.exe"


def write_abc_file(file_name, abc_strings):
    # Accept a single abc string
    if not isinstance(abc_strings, list):
        abc_strings = [abc_strings]

    # Create a valid abc string for writing to the output file
    file_string = ""
    for si in range(len(abc_strings)):
        file_string += "X:{}\nM:4/4\nK:Cmaj\n{}\n\n".format(si, abc_strings[si])

    # Write to a text file using abc notation
    with open(file_name, 'w') as file:
        file.write(file_string)


def generate_midi_file(abc_file_name, midi_file_name, tune_reference_number='',do_print=False):
    # Auto-correct file extension
    if midi_file_name[-4:] != ".mid":
        midi_file_name += ".mid"

    # Assert the required files exist
    assert os.path.exists(ABC_2_MIDI_PATH)
    assert os.path.exists(abc_file_name)

    # Run abc2midi.exe on the input file to create a midi file with given name
    command_string = '{} "{}" {} -o "{}"'.format(ABC_2_MIDI_PATH, abc_file_name, tune_reference_number, midi_file_name)
    output = os.popen(command_string)
    output = output.read()

    # Display console output
    if do_print:
        print(command_string)
        print(output)


def generate_ext_file(midi_file_name, pdf_file_name, file_extension="pdf", do_print=False):
    # Auto-correct midi file extension
    if midi_file_name[-4:] != ".mid":
        midi_file_name += ".mid"

    # Auto-correct pdf file extension
    if pdf_file_name[-4:] != "." + file_extension:
        pdf_file_name += "." + file_extension

    # Assert the required files exist
    assert os.path.exists(MUSESCORE_BIN_PATH)
    assert os.path.exists(midi_file_name)

    # Run musescore to generate pdf of the midi file
    command_string = '"{}" -o "{}" "{}"'.format(MUSESCORE_BIN_PATH, pdf_file_name, midi_file_name)
    output = os.popen(command_string)
    output = output.read()

    # Display console output
    if do_print:
        print(command_string)
        print(output)


def play_midi_file(midi_file_name, do_print=False):
    # Auto-correct file extension
    if midi_file_name[-4:] != ".mid":
        midi_file_name += ".mid"

    # Assert the required files exist
    assert os.path.exists(midi_file_name)

    # Open the midi file with os default program associated with midi
    command_string = midi_file_name
    output = os.popen(command_string)
    output = output.read()

    # Display console output
    if do_print:
        print(command_string)
        print(output)


# abc_string_list = ["|: A/2 G A d e B A | G E e E G e e d | c G E G c G E C | D E C B, A, A, G, A, | C E G 2 c 2",
#                    "|: A G A d e B A | G E e E G e e d | c G E G c G E C | D E C B, A, A, G, A, | C E G 2 c 2"]
# abc_file = "song.abc"
# midi_file = "song.mid"
# pdf_file = "song.pdf"
#
# write_abc_file(abc_file, abc_string_list)
# generate_midi_file(abc_file, midi_file, tune_reference_number=1, do_print=True)
# generate_pdf_file(midi_file, pdf_file, do_print=True)
