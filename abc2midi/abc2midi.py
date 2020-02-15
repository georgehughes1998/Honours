import os


def write_abc_file(file_name, abc_string):
    # Write to a text file using abc notation
    with open(file_name, 'w') as file:
        file_string = "X:1\nK:Cmaj\nM:4/4\n{}".format(abc_string)
        file.write(file_string)


def generate_midi_file(abc_file_name, midi_file_name):
    # Auto-correct file extension
    if midi_file_name[-4:] != ".mid":
        midi_file_name += ".mid"

    # Run abc2midi.exe on the input file to create a midi file with given name
    command_string = "abc2midi.exe {} -o {}".format(abc_file_name, midi_file_name)
    output = os.popen(command_string)
    output = output.read()

    # Return the output from abc2midi.exe
    return output


def play_midi_file(midi_file_name):
    # Auto-correct file extension
    if midi_file_name[-4:] != ".mid":
        midi_file_name += ".mid"

    # Open the midi file with os default program associated with midi
    command_string = midi_file_name
    output = os.popen(command_string)
