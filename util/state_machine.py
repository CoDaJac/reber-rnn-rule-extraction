import random as rnd
from functools import reduce


class StateMachine:
    def __init__(self, states):
        self.states = states
        self.alphabet = self.__get_alphabet()
    
    def __get_alphabet(self):
        transitions = reduce(lambda a, b: a + b, map(lambda s: s.transitions, self.states))
        return sorted(set(map(lambda t: t.value, transitions)))
              
    def run(self):
        current_state = next(s for s in self.states if s.is_initial)
        
        result = []
        while not current_state.is_final:
            rnd_idx = rnd.randrange(len(current_state.transitions))
            next_transition = current_state.transitions[rnd_idx]
            result.append(next_transition.value)
            current_state = next_transition.next_state
        
        return result
    
    def is_sequence_valid(self, sequence):
        current_state = next(s for s in self.states if s.is_initial)
        
        for c in sequence:
            try:
                index = list(t.value for t in current_state.transitions).index(c)
                current_state = current_state.transitions[index].next_state
            except ValueError:
                return False 
                 
        return True
            
        
class State:
    def __init__(self, name, is_initial=False, is_final=False):
        self.name = name
        self.is_initial = is_initial
        self.is_final = is_final
        self.transitions = []

class Transition:
    def __init__(self, value, next_state):
        self.value = value
        self.next_state = next_state

class ReberGrammarStateMachine(StateMachine):
    '''
    A finite state machine that generates sentences based on Reber's grammar.\n
    For further info see: http://wjh.harvard.edu/~pal/pdfs/pdfs/reber67.pdf
    '''
    MIN_LENGTH = 5
    MAX_LENGTH = 100
    
    def __init__(self):
        spre = State("Spre", is_initial=True)
        s0 = State("S0")
        s1 = State("S1")
        s2 = State("S2")
        s3 = State("S3")
        s4 = State("S4")
        sout = State("Sout")
        spost = State("Spost", is_final=True)
        
        spre.transitions = [
            Transition("#", s0)
        ]
        s0.transitions = [
            Transition("T", s1),
            Transition("V", s3)
        ]
        s1.transitions = [
            Transition("P", s1),
            Transition("T", s2)
        ]
        s2.transitions = [
            Transition("X", s3),
            Transition("S", sout)
        ]
        s3.transitions = [
            Transition("X", s3),
            Transition("V", s4)
        ]
        s4.transitions = [
            Transition("P", s2),
            Transition("S", sout)
        ]
        sout.transitions = [
            Transition("#", spost)
        ]
        
        super().__init__(states=[spre, s0, s1, s2, s3, s4, sout, spost])
