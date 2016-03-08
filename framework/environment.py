import numpy as np
from agent import *
from policy import *
from memory import *


class Environment(object):
    def __init__(self, name, rng=None, maxSteps = 30):
        self.name = name
        self.rng = rng
        if not self.rng:
            self.rng = np.random.RandomState()
        self.state = None
        self.isTerminal = False
        self.steps = 0
        self.maxSteps = maxSteps

    def doAction(self, action):
        raise NotImplementedError

    def step(self):
        self.steps += 1
        if self.maxSteps and self.steps > self.maxSteps:
            self.isTerminal = True

    def getState(self):
        return self.state[:].reshape((1,) + self.state.shape)

    def getNumActions(self):
        raise NotImplementedError

    def getValidActions(self):
        raise NotImplementedError

    def reset(self):
        self.isTerminal = False
        self.steps = 0

    def getTransition(self, actionNum):
        oldState = self.state[:].reshape((1,) + self.state.shape)
        action = self.getValidActions()[actionNum]
        reward = self.doAction(action)
        newState = self.state[:].reshape((1,) + self.state.shape)
        return Transition(oldState, actionNum, reward, newState, self.isTerminal)

# followed https://en.wikipedia.org/wiki/Mountain_Car for implementing the environment
# and Reinforcement Learning Benchmarks and Bake-offs II
class MountainCar(Environment):
    def __init__(self, rng=None, useWiki = False, maxSteps = 1000):
        super(MountainCar, self).__init__('MountainCar', rng, maxSteps)
        # state[0] is position, state[1] is velocity
        self.useWiki = useWiki
        self.reset()

    def doAction(self, action):
        super(MountainCar, self).step()
        # actions +1 - forward throttle, 0 - neutral, -1 - backward throttle
        self.state[1] += action * 0.001 - np.cos(3*self.state[0]) * 0.0025
        self.state[1] = np.max(self.state[1], -0.07)
        self.state[1] = np.min(self.state[1], 0.07)
        self.state[0] += self.state[1]
        self.state[0] =  np.max(self.state[0], -1.2)
        self.state[0] = np.min(self.state[0], 0.5)

        reward = -1
        if self.state[0] >= 0.5:
            self.isTerminal = True
            reward = 0

        return reward

    def getNumActions(self):
        return 3

    def getValidActions(self):
        return [-1, 0, 1]

    def reset(self):
        super(MountainCar, self).reset()
        self.state = np.array([self.rng.uniform(-1.1, 0.49), 0])

    def disp(self):
        return self.state
    

# http://brain.cc.kogakuin.ac.jp/~kanamaru/NN/CPRL/
# https://github.com/stober/cartpole/blob/master/src/__init__.py
class CartPole(Environment):
    def __init__(self, rng=None, continuous = False, maxSteps = 10000):
        super(CartPole, self).__init__('CartPole', rng, maxSteps)
        # state[0] is angle, state[1] is angular velocity, state[2] position, state[3] velocity
        self.continuous = continuous
        self.reset()
        self.force = 10.0
        self.g = 9.81
        self.massCart = 1.
        self.massPole = 0.1
        self.poleLength = 0.5
        self.muc = 0
        self.mup = 0
        self.timeStep = 0.02
        
    def reset(self):
        super(CartPole, self).reset()
        self.state = np.zeros(4)
        self.state[0] = self.rng.uniform(-np.pi/18., np.pi/18.)
        self.state[1] = 0
        self.state[2] = self.rng.uniform(-0.5, 0.5)
        self.state[3] = 0
        
    def getNumActions(self):
        if self.continuous:
            raise Error('not used for continuous cartpole')
        else:
            return 21

    def getValidActions(self):
        if self.continuous:
            raise Error('not used for continuous cartpole')
        else:
            return np.arange(21) - 10
    
    def doAction(self, action):
        super(CartPole, self).step()
        if not self.continuous:
            force = self.force * action / 10.
        else:
            force = action
            if abs(force) > self.force:
                force = np.sign(force) * self.force
                
        theta, thetaDot, x, xDot = self.state
        angularAcc = (self.g * np.sin(theta) \
                     + np.cos(theta) / (self.massCart + self.massPole) \
                     * (self.muc * np.sign(xDot) - force - self.massPole * self.poleLength * thetaDot ** 2 * np.sin(theta)) \
                     - self.mup * thetaDot / (self.massPole * self.poleLength)) \
                     * self.poleLength * (3./4. - self.massPole / (self.massPole + self.massCart) * (np.cos(theta) ** 2))
        xAcc = (force + self.massPole * self.poleLength * \
               (thetaDot ** 2 * np.sin(theta) - angularAcc * np.cos(theta)) \
               - self.muc * theta / (self.massPole * self.poleLength)) \
               / (self.massPole + self.massPole)
            
        thetaDot += self.timeStep * angularAcc
        theta += self.timeStep * thetaDot
        xDot += self.timeStep * xAcc
        x += self.timeStep * xDot
        
        self.state = np.array([theta, thetaDot, x, xDot])
        
        reward = -1
        if abs(x) > 2.4 or abs(theta) > np.pi / 6.:
            reward = -1000
            self.isTerminal = True
        elif -0.05 < x < 0.05 and abs(theta) < np.pi / 60.:
            reward = 0
        return reward
    
    def disp(self):
        return self.state
        

class GridWorld(Environment):
    def __init__(self, maxSteps = 1000, mode = 'allrand', size = 4, rng = None):
        super(GridWorld, self).__init__('GridWorld', rng, maxSteps)
        self.size = size
        self.mode = mode
        self.reset()

    def buildGrid(self):
        self.grid = np.zeros((4, self.size, self.size))
        self.grid[0][self.playerLoc] = 1
        self.grid[1][self.pitLoc] = 1
        self.grid[2][self.goalLoc] = 1
        self.grid[3][self.wallLoc] = 1
        self.state = self.grid

    def generateExampleGrid(self):
        self.playerLoc = (0,1)
        self.pitLoc = (1,1)
        self.goalLoc = (3,3)
        self.wallLoc = (2,2)

    def generateRandPlayerGrid(self):
        self.playerLoc = (self.rng.randint(self.size),self.rng.randint(self.size))
        self.pitLoc = (1,1)
        self.goalLoc = (1,2)
        self.wallLoc = (2,2)

        self.buildGrid()
        if (np.sum(self.grid, axis = 0) > 1).any():
            self.generateRandPlayerGrid()

    def generateRandGrid(self):
        self.playerLoc = (self.rng.randint(self.size),self.rng.randint(self.size))
        self.pitLoc = (self.rng.randint(self.size),self.rng.randint(self.size))
        self.goalLoc = (self.rng.randint(self.size),self.rng.randint(self.size))
        self.wallLoc = (self.rng.randint(self.size),self.rng.randint(self.size))

        self.buildGrid()
        if (np.sum(self.grid, axis = 0) > 1).any():
            self.generateRandGrid()

    def doAction(self, action):
        super(GridWorld, self).step()
        if action == 0: # go north
            newLoc = (self.playerLoc[0] - 1, self.playerLoc[1])
            if newLoc != self.wallLoc and (np.array(newLoc) < (self.size, self.size)).all() and \
                    (np.array(newLoc) > (-1, -1)).all():
                self.playerLoc = newLoc
        elif action == 1: # go south
            newLoc = (self.playerLoc[0] + 1, self.playerLoc[1])
            if newLoc != self.wallLoc and (np.array(newLoc) < (self.size, self.size)).all() and \
                    (np.array(newLoc) > (-1, -1)).all():
                self.playerLoc = newLoc
        elif action == 2: # go west
            newLoc = (self.playerLoc[0], self.playerLoc[1] - 1)
            if newLoc != self.wallLoc and (np.array(newLoc) < (self.size, self.size)).all() and \
                    (np.array(newLoc) > (-1, -1)).all():
                self.playerLoc = newLoc
        else: # go east
            newLoc = (self.playerLoc[0], self.playerLoc[1] +1)
            if newLoc != self.wallLoc and (np.array(newLoc) < (self.size, self.size)).all() and \
                    (np.array(newLoc) > (-1, -1)).all():
                self.playerLoc = newLoc
        self.buildGrid()
        reward = -1
        if self.playerLoc == self.pitLoc:
            reward = -10
        elif self.playerLoc == self.goalLoc:
            reward = 10
        if reward != -1:
            self.isTerminal = True
        return reward

    def disp(self):
        disp = np.zeros((self.size, self.size))
        disp[self.wallLoc] = 1
        disp[self.pitLoc] = 2
        disp[self.goalLoc] = 3
        disp[self.playerLoc] = 4

        return str(disp).replace('1', 'W').replace('0', ' ').replace('2', 'O').replace('3', 'G').replace('4', 'P').replace('.', '')

    def reset(self):
        super(GridWorld, self).reset()
        self.grid = np.zeros((4, self.size, self.size))
        self.playerLoc = (0,0)
        self.pitLoc = (0,0)
        self.goalLoc = (0,0)
        self.wallLoc = (0,0)

        if self.mode == 'allrand':
            self.generateRandGrid()
        elif self.mode == 'prand':
            self.generateRandPlayerGrid()
        else:
            self.generateExampleGrid()
        self.buildGrid()

    def getNumActions(self):
        return 4

    def getValidActions(self):
        return [0, 1, 2, 3]
