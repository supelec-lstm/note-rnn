# Import pychain
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '../../pychain/src')))

# STL
import pickle
import time
from datetime import datetime
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
from optimization_algorithm import *

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
pitch_to_index = {pitch: nb_special_tokens + i for i, pitch in enumerate(pitches)}
# Durations
durations = sorted(list({note.duration.quarterLength for note in all_notes}))
nb_durations = len(durations)
duration_to_index = {duration: i for i, duration in enumerate(durations)}
# Create the dataset
dataset = []
for part in opus:
    # Remove rests and chords
    notes = part.flat.getElementsByClass(note.Note)
    dataset += [(0, 0)] + \
        [(pitch_to_index[note.midi], duration_to_index[note.duration.quarterLength]) for note in notes]
nb_features = nb_special_tokens + nb_pitches + nb_durations
# Clean
del all_notes
del opus

# Parameters of the network
num_lstms = 2
dim_s = 128
learning_rate = 2e-3
len_seq = 50
nb_seq_per_batch = 50
hidden_shapes = [(nb_seq_per_batch, dim_s), (nb_seq_per_batch, dim_s)] * num_lstms

# Learn

def notes_to_sequences(tokens, nb_seq, len_seq):
    sequences = np.zeros((len_seq, nb_seq, nb_features))
    for i_seq in range(nb_seq):
        for i, token in enumerate(tokens[i_seq*len_seq:(i_seq+1)*len_seq]):
            #print(i, i_seq)
            sequences[i, i_seq, token[0]] = 1
            if token[0] >= nb_special_tokens:
                sequences[i, i_seq, nb_special_tokens + nb_pitches + token[1]] = 1
    return sequences

def learn(layer):
    # Create the graph
    graph = RecurrentGraph(layer, len_seq - 1, hidden_shapes)
    # Optimization algorithm
    #algo = GradientDescent(graph.get_learnable_nodes(), learning_rate)
    algo = RMSProp(graph.get_learnable_nodes(), learning_rate)
    # Learn
    i_pass = 1
    i_batch = 1
    nb_batches = int(len(dataset) / nb_seq_per_batch / len_seq)
    print(len(dataset))
    while True:
        # Shuffle sequences
        #np.random.shuffle(dataset)
        for i in range(0, len(dataset) // len_seq, nb_seq_per_batch):
            t_start = time.time()
            # Take a new batch
            notes = dataset[i*len_seq:(i+nb_seq_per_batch)*len_seq]
            sequences = notes_to_sequences(notes, nb_seq_per_batch, len_seq)
            # Propagate and backpropagate the batch
            output = graph.propagate(sequences[:-1])
            cost = graph.backpropagate(sequences[1:]) / len_seq / nb_seq_per_batch
            # Descend gradient
            algo.optimize(nb_seq_per_batch)
            # Print info
            print('pass: ' + str(i_pass) + ', batch: ' + str((i_batch-1)%nb_batches+1) + '/' + str(nb_batches) + \
                ', cost: ' + str(cost) + ', time: ' + str(time.time() - t_start))
            # Save
            if i_batch % 1000 == 0:
                save_layer(layer, i_batch)
            i_batch += 1
        i_pass += 1

def create_layer():
    # Input
    x = InputNode()
    hidden_inputs = []
    hidden_outputs = []
    lstms = []
    # LSTMs
    parent = x
    for i in range(num_lstms):
        h_in = InputNode()
        s_in = InputNode()
        dim_x = dim_s
        if i == 0:
            dim_x = nb_features
        lstm = LSTMWFGNode(dim_x, dim_s, [parent, h_in, s_in])
        h_out = IdentityNode([(lstm, 0)])
        s_out = IdentityNode([(lstm, 1)])
        parent = h_out
        # Add to containers
        hidden_inputs += [h_in, s_in]
        hidden_outputs += [h_out, s_out]
        lstms.append(lstm)
    # Outputs
    w = LearnableNode(0.1 * np.random.randn(dim_s, nb_features))
    mult = MultiplicationNode([parent, w])
    y = InputNode()
    # Pitch or special token
    select_pitch = SelectionNode([mult], 0, nb_special_tokens + nb_pitches)
    out_pitch = SoftmaxNode([select_pitch])
    y_pitch = SelectionNode([y], 0, nb_special_tokens + nb_pitches)
    cost_pitch = SoftmaxCrossEntropyNode([y_pitch, out_pitch])
    # Duration
    select_duration = SelectionNode([mult], nb_special_tokens + nb_pitches, nb_features)
    out_duration = SoftmaxNode([select_duration])
    y_duration = SelectionNode([y], nb_special_tokens + nb_pitches, nb_features)
    cost_duration = SoftmaxCrossEntropyNode([y_duration, out_duration])
    # Final output
    out = ConcatenationNode([out_pitch, out_duration])
    # Total Cost
    cost = AdditionNode([cost_pitch, cost_duration])

    nodes = hidden_inputs + hidden_outputs + lstms + [x, w, mult, y, \
        select_pitch, out_pitch, y_pitch, cost_pitch, \
        select_duration, out_duration, y_duration, cost_duration, 
        out, cost]
    return Layer(nodes, [x], [out], hidden_inputs, hidden_outputs, [y], cost, [w] + lstms)

def save_layer(layer, i_batch):
    path = 'models/' + str(datetime.now()) + '_b:' +  str(i_batch) + '.pickle'
    pickle.dump(layer, open(path, 'wb'))

# Main

if __name__ == '__main__':
    layer = create_layer()
    nb_weights = 0
    for node in layer.get_learnable_nodes():
        nb_weights += node.w.shape[0]*node.w.shape[1]
    print('number of parameters in the model:', nb_weights)
    learn(layer)