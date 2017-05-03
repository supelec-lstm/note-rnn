# Import pychain
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../pychain/src')))

# STL
import pickle
# matplotlib
import matplotlib.pyplot as plt
# music21
from music21 import *
# To set in order to export as midi or score
environment.set('musicxmlPath', '/usr/bin/musescore')
environment.set('midiPath', '/usr/bin/timidity')
# pychain
from lstm_node import *
from layer import *
from recurrent_graph import *

# Read file
path = 'jigs.abc'
opus = converter.parse(path)

# Dataset
all_notes = opus.flat.getElementsByClass(note.Note)
# Special tokens
nb_special_tokens = 1 # to signal a new part
# Pitches
pitches = sorted(list({note.midi for note in all_notes}))
nb_pitches = len(pitches)
index_to_pitch = {nb_special_tokens + i: pitch for i, pitch in enumerate(pitches)}
# Durations
durations = sorted(list({note.duration.quarterLength for note in all_notes}))
nb_durations = len(durations)
index_to_duration = {i: duration for i, duration in enumerate(durations)}
nb_features = nb_special_tokens + nb_pitches + nb_durations
# Clean
del all_notes
del opus

num_lstms = 2
dim_s = 128
len_seq = 300
hidden_shapes = [(1, dim_s), (1, dim_s)] * num_lstms

def stochastic(output):
    chosen_note = np.zeros((1, nb_features))
    # Choose pitch
    chosen_pitch = np.random.choice(nb_special_tokens + nb_pitches, p=output.flatten()[:nb_special_tokens + nb_pitches])
    chosen_note[0,chosen_pitch] = 1
    if chosen_pitch >= nb_special_tokens:
        chosen_duration = np.random.choice(nb_durations, p=output.flatten()[nb_special_tokens + nb_pitches:nb_features])
        chosen_note[0,nb_special_tokens + nb_pitches + chosen_duration] = 1
    # Choose duration
    return chosen_note

def sequence_to_tokens(sequence):
    tokens = []
    for y in sequence:
        chosen_pitch = np.argmax(y[0,0:nb_special_tokens + nb_pitches])
        if chosen_pitch >= nb_special_tokens:
            chosen_duration = np.argmax(y[0,nb_special_tokens + nb_pitches:nb_features])
        else:
            chosen_duration = 0
        tokens.append((chosen_pitch, chosen_duration))
    return tokens

def tokens_to_opus(tokens):
    opus = stream.Opus()
    cur_score = stream.Score()
    for token in tokens:
        if token[0] == 0:
            opus.append(cur_score)
            cur_score = stream.Score()
        elif token[0] >= nb_special_tokens:
            n = note.Note()
            n.midi = index_to_pitch[token[0]]
            n.duration.quarterLength = index_to_duration[token[1]]
            cur_score.append(n)
    opus.append(cur_score)
    opus.show('text')
    opus.show('midi')

def sample(graph):
    x = np.zeros((1, nb_features))
    x[0] = 1
    result = graph.generate(stochastic, x)
    #print(result)
    tokens = sequence_to_tokens(result)
    #print(str((0, 0)) + str(tokens))
    tokens_to_opus(tokens)

if __name__ == '__main__':
    # To change with the correct model
    layer = pickle.load(open('models/2017-05-03 18:14:32.017278_b:6000.pickle', 'rb'))
    graph = RecurrentGraph(layer, len_seq - 1, hidden_shapes)
    sample(graph)