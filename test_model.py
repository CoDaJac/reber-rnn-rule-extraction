import numpy as np
from keras.models import load_model

from settings import *
from util.encoder import *
from util.state_machine import ReberGrammarStateMachine

# Load the model
model = load_model("./model/reber-rnn-model.h5")
model.load_weights("./model/reber-rnn-weights.hdf5")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Set a start pattern
pattern = [char_to_int["#"], char_to_int["T"]]

# Generate sequence based on predictions
result = "".join(list(int_to_char[c] for c in pattern))
for i in range(100): 
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(ALPHABET))
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result += int_to_char[index]
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
    if(int_to_char[index] == "#"): break
    
print(f"Result '{result}' is{'' if ReberGrammarStateMachine().is_sequence_valid(result) else ' not'} valid with Reber's grammar")