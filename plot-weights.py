import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.utils import np_utils
from numpy import savetxt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Embedding, SimpleRNN, TimeDistributed)
from tensorflow.keras.models import Sequential

import util.reber_grammar_generator as reber
from settings import *
from util.encoder import *


def get_weights_at_epoch(epoch):
    print(f"Start getting weights step-by-step at epoch {epoch} with seed={SEED}...")

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

    # Reshape X to be [samples, time steps, features] and normalize
    enc_X = np.reshape(data_X, (len(data_X), PATTERN_LENGTH, 1))
    enc_X = enc_X / float(len(ALPHABET))

    # One-hot encode the output variables
    enc_Y = np_utils.to_categorical(data_Y)

    # Define RNN/LSTM
    model = Sequential([
        LSTM(NUM_NEURONS, input_shape=(enc_X.shape[1], enc_X.shape[2]), return_sequences=True),
        Dropout(DROPOUT_RATE),
        LSTM(NUM_NEURONS),
        Dropout(DROPOUT_RATE),
        Dense(units=enc_Y.shape[1], activation='softmax')
    ])

    # Fit the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    if(epoch > 1):
        print(f"Training model up to epoch {epoch-1}")
        model.fit(enc_X, enc_Y, epochs=epoch-1, batch_size=5)

    # Get lstm layer
    lstm = next(l for l in model.layers if l.name == "lstm")
    
    # Step-by-Step training
    print("Starting to train model step-by-step...")
    weights = [[] for i in range(0, NUM_NEURONS)]
    for i in range(0, len(enc_X)):
        ex = np.array([enc_X[i]])
        ey = np.array([enc_Y[i]])
        model.fit(ex, ey, epochs=1, batch_size=1, verbose=1)

        for i in range(0, len(lstm.get_weights()[1])):
            weights[i].extend(lstm.get_weights()[1][i])
            # weights[i].append(np.max(lstm.get_weights()[1][i]))

    # Print model score
    score = model.evaluate(x=enc_X, y=enc_Y, verbose=1)
    print(model.metrics_names)
    print(score)
    
    # Normalize weights
    minimum = np.min([np.min(w) for w in weights])
    maximum = np.max([np.max(w) for w in weights])
    weights = np.array([((np.array(x) - minimum) / (maximum - minimum)) for x in weights])
    # weights = np.array([np.array(x) / float(max(x, key=abs)) for x in weights])
    print(weights)
    return weights

def main():
    weights = get_weights_at_epoch(EPOCH)
    savetxt(f"./plot_weights/{int(time.time())}-e{EPOCH}-x{WEIGHT_X_IDX}-y{WEIGHT_Y_IDX}-seed{SEED}.csv", weights, delimiter=";")
    
    plt.plot(weights[WEIGHT_X_IDX], weights[WEIGHT_Y_IDX], "bo", markersize=1)
    plt.title(f"epoch={EPOCH}")
    plt.xlabel(f"s{WEIGHT_X_IDX}")
    plt.ylabel(f"s{WEIGHT_Y_IDX}")
    plt.show()
    
if __name__ == "__main__":
    main()
