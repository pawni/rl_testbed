import numpy as np

class Policy(object):
    def __init__(self, name, numActions, rng = None):
        self.name = name
        self.rng = rng
        self.numActions = numActions
        if not self.rng:
            self.rng = np.random.RandomState()
            
    def getAction(self, state):
        raise NotImplementedError
        
# TODO: implement decay modular, currently:exponential
class eGreedy(Policy):
    def __init__(self, eps, numActions, epsDecay = None, epsMin = 0.1, epochs = None, rng = None):
        super(eGreedy, self).__init__('eGreedy', numActions, rng)
        self.eps = eps
        self.epsMin = epsMin
        self.epochs = epochs
        self.epsDecay = epsDecay
        self.numActions = numActions
        
    def getAction(self, state):
        action = None
        if self.rng.rand() < self.eps:
            action = self.rng.randint(0, self.numActions)
        return action
    
    def decay(self):
        if self.eps > self.epsMin:
            if self.epochs:
                self.eps -= 1./self.epochs
            elif self.epsDecay:
                self.eps *= epsDecay
            self.eps = np.max(self.epsMin, self.eps)
        
    
class Greedy(Policy):
    def __init__(self, numActions):
        super(Greedy, self).__init__('Greedy', numActions)
        
    def getAction(self, state):
        pass