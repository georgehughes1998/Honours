import os


# Lib-level constants
MUSESCORE_BIN_PATH = "C:/Program Files (x86)/MuseScore 2/bin/MuseScore.exe"
OUTPUT_PATH = "output/"


def write_abc_file(file_name, abc_string):
    # Write to designated output dir
    file_name = OUTPUT_PATH + file_name

    # Write to a text file using abc notation
    with open(file_name, 'w') as file:
        file_string = "X:1\nK:Cmaj\nM:4/4\n{}".format(abc_string)
        file.write(file_string)


def generate_midi_file(abc_file_name, midi_file_name, do_print=False):
    # Auto-correct file extension
    if midi_file_name[-4:] != ".mid":
        midi_file_name += ".mid"

    # Write to designated output dir
    abc_file_name = OUTPUT_PATH + abc_file_name
    midi_file_name = OUTPUT_PATH + midi_file_name

    # Assert the required files exist
    assert os.path.exists(abc_file_name)

    # Run abc2midi.exe on the input file to create a midi file with given name
    command_string = "abc2midi.exe {} -o {}".format(abc_file_name, midi_file_name)
    output = os.popen(command_string)

    # Display console output
    if do_print:
        output = output.read()
        print("abc2midi.exe output:", output)


def generate_pdf_file(midi_file_name, pdf_file_name, do_print=False):
    # Auto-correct midi file extension
    if midi_file_name[-4:] != ".mid":
        midi_file_name += ".mid"

    # Auto-correct pdf file extension
    if pdf_file_name[-4:] != ".pdf":
        pdf_file_name += ".pdf"

    # Write to designated output dir
    pdf_file_name = OUTPUT_PATH + pdf_file_name
    midi_file_name = OUTPUT_PATH + midi_file_name

    # Assert the required files exist
    assert os.path.exists(MUSESCORE_BIN_PATH)
    assert os.path.exists(midi_file_name)

    # Run musescore to generate pdf of the midi file
    command_string = '"{}" -o "{}" "{}"'.format(MUSESCORE_BIN_PATH, pdf_file_name, midi_file_name)
    output = os.popen(command_string)

    # Display console output
    if do_print:
        output = output.read()
        print("MuseScore.exe output:", output)


def play_midi_file(midi_file_name, do_print=False):
    # Auto-correct file extension
    if midi_file_name[-4:] != ".mid":
        midi_file_name += ".mid"

    # Write to designated output dir
    midi_file_name = OUTPUT_PATH + midi_file_name

    # Assert the required files exist
    assert os.path.exists(midi_file_name)

    # Open the midi file with os default program associated with midi
    command_string = midi_file_name
    output = os.popen(command_string)

    # Display console output
    if do_print:
        output = output.read()
        print("Default midi program output:", output)


abc_string = "|: A/2 G A d e B A | G E e E G e e d | c G E G c G E C | D E C B, A, A, G, A, | C E G 2 c 2"
abc_file = "song.abc"
midi_file = "song.mid"
pdf_file = "song.pdf"

write_abc_file(abc_file, abc_string)
generate_midi_file(abc_file, midi_file, do_print=True)
generate_pdf_file(midi_file, pdf_file, do_print=True)
