from util.state_machine import ReberGrammarStateMachine
import random as rnd

# ============================== Model generation ===========================
ALPHABET = list(ReberGrammarStateMachine().alphabet)    # Reber alphabet
SEED = 174845               # Random seed
SEQ_MIN_LENGTH = 5          # Minimum length of sequences
SEQ_MAX_LENGTH = 10         # Maximum length of sequences 
NUM_EXAMPLES = 100          # Number of positive examples
PATTERN_LENGTH = 2          # Length of a single pattern in data_x
TEST_SIZE = 0.1             # Ratio of test-set for the split
NUM_NEURONS = 2             # The number of lstm-neurons in the model
DROPOUT_RATE = 0.2          # The dropout rate of a dropout layer
EPOCH = 1                 # The number of epochs to train
# ===========================================================================  

# ============================== Weight plotting ===========================
WEIGHT_X_IDX = 0                    # The index of the neuron that is plotted on the x-axis
WEIGHT_Y_IDX = NUM_NEURONS - 1      # The index of the neuron that is plotted on the y-axis
# ===========================================================================  