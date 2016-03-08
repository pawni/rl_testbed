import numpy as np

class Transition(object):
    def __init__(self, oldState, action, reward, newState, terminal):
        self.oldState = oldState
        self.action = action
        self.newState = newState
        self.reward = reward
        self.terminal = terminal
        
class Memory(object):
    def __init__(self, length, sampleSize, rng = None, name = 'memory'):
        self.name = name
        self.length = length
        self.sampleSize = sampleSize
        self.rng = rng
        if not self.rng:
            self.rng = np.random.RandomState()
        self.memory = []
        self.lastInsert = 0
            
    def addTransition(self, transition):
        if len(self.memory) < self.length:
            self.memory.append(transition)
        else:
            self.memory[self.lastInsert] = transition
            self.lastInsert += 1
            if self.lastInsert == self.length:
                self.lastInsert = 0
        
    def sampleTransitions(self):
        if len(self.memory) < self.sampleSize:
            raise Error
        return self.rng.choice(self.memory, self.sampleSize)
    
    def isReady(self):
        if len(self.memory) < self.sampleSize:
            return False
        return True
    
           