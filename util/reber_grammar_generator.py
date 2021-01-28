import random as rnd
import sys

from util.state_machine import ReberGrammarStateMachine


def generate_sequences(quantity, seed=None):
    '''
    Returns a set of &lt;quantity&gt; randomly generated sentences that are valid with Reber's grammar.\n
    Randomization can optionally be determined by setting &lt;seed&gt;.
    '''
    assert(quantity > 0)
    assert(quantity <= 1000)

    if (seed != None): rnd.seed(seed)
    state_machine = ReberGrammarStateMachine()
    
    result = set()
    while len(result) < quantity:
        sequence = state_machine.run()
        result.add("".join(sequence))
          
    return result

def generate_sequences_of_length(quantity, min_length, max_length, seed=None):
    '''
    Returns a set of &lt;quantity&gt; randomly generated sentence of &lt;max_length&gt; <= length >= &lt;min_length&gt; that is valid with Reber's grammar.\n
    Randomization can optionally be determined by setting &lt;seed&gt;.
    '''
    assert(min_length >= ReberGrammarStateMachine.MIN_LENGTH)
    assert(max_length <= ReberGrammarStateMachine.MAX_LENGTH)
    assert(quantity > 0)
    assert(quantity <= 1000)

    if (seed != None): rnd.seed(seed) 
    state_machine = ReberGrammarStateMachine()
    
    result = list()
    while(len(result) < quantity):
        sequence = state_machine.run()
        
        if(len(sequence) >= min_length and len(sequence) <= max_length):
            result.append("".join(sequence))
          
    return result

def generate_non_reber_sequences_of_length(quantity, min_length, max_length, seed=None):
    '''
    Returns a set of &lt;quantity&gt; randomly generated sentences of &lt;max_length&gt; <= length >= &lt;min_length&gt; that are not valid with Reber's grammar while using a valid alphabet.\n
    Randomization can optionally be determined by setting &lt;seed&gt;.
    '''
    assert(min_length >= ReberGrammarStateMachine.MIN_LENGTH)
    assert(max_length <= ReberGrammarStateMachine.MAX_LENGTH)
    assert(quantity > 0)
    assert(quantity <= 1000)

    if(seed != None): rnd.seed(seed)
    state_machine = ReberGrammarStateMachine()
    
    result = set()
    while(len(result) < quantity):
        sequence_len = min_length if min_length==max_length else rnd.randrange(min_length, max_length)
        sequence = "".join(rnd.choice(state_machine.alphabet) for i in range(sequence_len))
        
        if(not state_machine.is_sequence_valid(sequence)):
            result.add(sequence)
          
    return result
    