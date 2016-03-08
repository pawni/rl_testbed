import numpy as np
from environment import *
from policy import *
from memory import *

class Agent(object):
    def __init__(self, name, policy = None, memory = None, rng = None):
        self.name = name
        self.rng = rng
        if not self.rng:
            self.rng = np.random.RandomState()
        self.policy = policy
        self.state = None
        self.isTerminal = False
        self.memory = memory

    def getBestAction(self, state):
        raise NotImplementedError

    def train(self, environment, epochs = 100):
        rewards = np.zeros(epochs)
        for epoch in xrange(epochs):
            print 'at epoch %d' % epoch
            environment.reset()
            rew = 0
            i = 0
            while not environment.isTerminal:
                #print 'at step %d' % i,
                curState = environment.getState()
                action = None
                if self.policy:
                    action = self.policy.getAction(curState)
                if action == None:
                    action = self.getBestAction(curState)
                trans = environment.getTransition(action)
                if self.memory:
                    self.memory.addTransition(trans)
                    if self.memory.isReady():
                        samples = self.memory.sampleTransitions()
                        self.learnBatch(samples)
                else:
                    self.learn(trans)
                rew += trans.reward
                i += 1.
            rewards[epoch] = rew / i
            self.policy.decay()
        return rewards

    def evaluate(self, environment):
        environment.reset()
        while not environment.isTerminal:
            curState = environment.getState()
            print environment.disp()
            print self.network.predict(curState).flatten()
            action = self.getBestAction(curState)
            print action
            trans = environment.getTransition(environment.getValidActions()[action])
            print trans.reward
        print environment.disp()

    def test(self, environmentList):
        raise NotImplementedError
        
    def learn(self, trans):
        raise NotImplementedError
        
    def learnBatch(self, samples):
        raise NotImplementedError

# http://outlace.com/Reinforcement-Learning-Part-3/
class DQN(Agent):
    def __init__(self, network, gamma = 0.9, policy = None, memory = None, rng = None):
        super(DQN, self).__init__('DQN', policy, memory, rng)
        self.network = network
        self.gamma = gamma

    def getBestAction(self, state):
        qs = self.network.predict(state).flatten()
        maxQ = qs[0]
        bestAction = [0]


        for i, q in enumerate(qs):
            if q > maxQ:
                maxQ = [q]
                bestAction = [i]
            elif q == maxQ:
                bestAction.append(i)
        idx = self.rng.randint(len(bestAction))
        return bestAction[idx]

    def learn(self, transition):
        '''
        transition.oldState
        transition.newState
        transition.action
        transition.reward
        transition.terminal
        '''
        oldQs = self.network.predict(transition.oldState).flatten()
        newQs = self.network.predict(transition.newState).flatten()
        target = oldQs[:]
        target[transition.action] = transition.reward + (1 - transition.terminal) * (self.gamma * np.max(newQs))
        self.network.fit(transition.oldState, target.reshape(1, -1), nb_epoch=1, verbose = False)
        
    def learnBatch(self, samples):
        trainData = np.zeros((len(samples),) + samples[0].oldState.shape[1:])
        targets = None
        for i, transition in enumerate(samples):
            oldQs = self.network.predict(transition.oldState).flatten()
            newQs = self.network.predict(transition.newState).flatten()
            target = oldQs[:]
            target[transition.action] = transition.reward + (1 - transition.terminal) * (self.gamma * np.max(newQs))
            if type(targets) == type(None):
                targets = np.zeros((len(samples), ) + target.shape)
            targets[i] = target
            trainData[i] = transition.oldState
        
        self.network.fit(trainData, targets, nb_epoch=1, verbose = False)
