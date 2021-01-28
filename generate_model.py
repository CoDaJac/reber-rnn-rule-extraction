import sys

import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN, Embedding, TimeDistributed
from tensorflow.keras.models import Sequential

import util.reber_grammar_generator as reber
from settings import *
from util.encoder import *

print("Start generating model...") 

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

# Stratified training-test-split 
(train_X, test_X, train_Y, test_Y) = train_test_split(enc_X, enc_Y, test_size=TEST_SIZE, random_state=SEED, stratify=enc_Y)

# Define RNN/LSTM
model = Sequential([
    LSTM(NUM_NEURONS, input_shape=(enc_X.shape[1], enc_X.shape[2]), return_sequences=True),
    Dropout(DROPOUT_RATE),
    LSTM(NUM_NEURONS),
    Dropout(DROPOUT_RATE),
    Dense(units=enc_Y.shape[1], activation='softmax')
])

# Define a checkpoint
checkpoint = ModelCheckpoint("./model/reber-rnn-weights.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Fit the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(enc_X, enc_Y, epochs=EPOCH, batch_size=5, validation_data=(test_X, test_Y), callbacks=callbacks_list)

# Evaluate the model
score = model.evaluate(x=test_X, y=test_Y, verbose=1)
print(model.metrics_names)
print(score)

# Save the model
model.save("./model/reber-rnn-model.h5")

print("Done generating model...")
