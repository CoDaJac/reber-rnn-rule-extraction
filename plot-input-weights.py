import time

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from numpy import savetxt

import util.reber_grammar_generator as reber
from settings import *
from util.encoder import *
from util.state_machine import ReberGrammarStateMachine


def get_input_weights():
    print("Starting to generate weight matrix...")
    
    # Load the model
    model = load_model("./model/reber-rnn-model.h5")
    model.load_weights("./model/reber-rnn-weights.hdf5")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Generate example data
    examples = reber.generate_sequences_of_length(NUM_EXAMPLES, SEQ_MIN_LENGTH, SEQ_MAX_LENGTH, SEED)

    # Prepare the dataset of input to output pairs encoded as integers
    data_X = []
    data_Y = []
    for seq in examples:
        for i in range(0, len(seq) - PATTERN_LENGTH):
            seq_in = seq[i:i + PATTERN_LENGTH]
            seq_out = seq[i + PATTERN_LENGTH]
            data_X.append([char_to_int[char] for char in seq_in])
            data_Y.append(char_to_int[seq_out])
            
    # Remove loops (which cause a heavy bias)
    # TODO: Improve
    length = len(data_X)   
    for i in range(length-1, -1, -1):
        if(len(set(data_X[i])) == 1 and data_Y[i] == data_X[i][0]):
            for j in range(i-1, -1, -1):
                if(set(data_X[i]) == set(data_X[j]) and data_Y[i] == data_Y[j]):
                    del data_X[i]
                    del data_Y[i]
                    length -= 1
                    break
            
    # Generate weights based on predictions
    initial_weights = next(l.get_weights()[1][-1] for l in model.layers if l.name == "lstm")
    weights = [[np.max(initial_weights)], []]
    chars = []
    for p in data_X:
        pattern = p
        for i in range(0, 100):
            x = np.reshape(pattern, (1, len(pattern), 1))
            x = x / float(len(ALPHABET))
            prediction = model.predict(x, verbose=0)
            index = np.argmax(prediction)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]
            
            # weights[0].append(np.max(prediction))
            # weights[1].append(np.max(prediction))
            # weights[0].append(np.max(prediction))
            # weights[1].append(sorted(prediction[0])[-2])
            
            weights[0].extend(prediction[0])
            weights[1].extend(prediction[0])
            chars.extend(int_to_char[index])
            
            if(int_to_char[index] == "#"): break
        
    weights[0] = weights[0][:-1]
    weights = np.array([np.array(x) / float(max(x, key=abs)) for x in weights])
    
    print(weights)
    print(chars)
    return (weights, chars)

def generate_equipart_plot():
    (weights, chars) = get_input_weights()
    curr_time = int(time.time())
    savetxt(f"./input_weights/weights-{curr_time}.csv", weights, delimiter=";")
    savetxt(f"./input_weights/chars-{curr_time}.csv", chars, delimiter=";", fmt='%s')
    
    plt.plot(weights[0], weights[1], "bo", markersize=2)
    plt.plot([0.33, 0.33], [0, 1], "k-", linewidth=0.5)
    plt.plot([0.66, 0.66], [0, 1], "k-", linewidth=0.5)
    plt.plot([0, 1], [0.33, 0.33], "k-", linewidth=0.5)
    plt.plot([0, 1], [0.66, 0.66], "k-", linewidth=0.5)
    plt.xlabel("s0")
    plt.ylabel("s1")
    axes = plt.gca()
    axes.set_xlim([-0.01,1.01])
    axes.set_ylim([-0.01,1.01])
    plt.show()

def main():
    generate_equipart_plot()
    
if __name__ == "__main__":
    main()
