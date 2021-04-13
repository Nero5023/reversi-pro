import math

# W: total weight
# N: number visit
#

from board import BOARD_SIZE_LEN
import numpy as np
from collections import defaultdict


class MCTSNode:
    def __init__(self, state, move, parent=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.is_expanded = False
        self.children = {}
        self.parent = parent
        # +1 for pass move
        self.child_priors = np.zeros([BOARD_SIZE_LEN+1], dtype=np.float32)
        self.child_total_values = np.zeros([BOARD_SIZE_LEN+1], dtype=np.float32)
        self.child_number_visits = np.zeros([BOARD_SIZE_LEN+1], dtype=np.float32)

    @property
    def N(self):
        return self.parent.child_number_visits[self.move]

    @N.setter
    def N(self, value):
        self.parent.child_number_visits[self.move] = value

    @property
    def W(self):
        return self.parent.child_total_values[self.move]

    @W.setter
    def W(self, value):
        self.parent.child_total_values[self.move] = value

    def child_Q(self):
        # return self.child_total_values / (1+self.child_number_visits)
        return self.child_total_values / (self.child_number_visits + (self.child_number_visits == 0))

    def child_U(self):
        return math.sqrt(self.N) * (self.child_priors / (1 + self.child_number_visits))

    def best_child(self):
        return np.argmax(self.child_Q() + self.child_U())

    def select_leaf(self):
        node = self
        while node.is_expanded:
            action = node.best_child()
            node = node.maybe_add_child(action)
        return node

    def maybe_add_child(self, action):
        if action not in self.children:
            self.children[action] = MCTSNode(state=self.state.take_move(action), move=action, parent=self)
        return self.children[action]

    def expand(self, child_prior_probabilities):
        if self.is_expanded:
            return
        self.child_priors = child_prior_probabilities

    def back_update(self, value):
        node = self
        # TODO: 用 game 里面的 who player lai 确定正负
        factor = 1
        # check if node is root
        while node.parent is not None:
            node.N += 1
            node.W += (value*factor)
            node = node.parent
            factor = factor * -1


class SentinelNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_values = defaultdict(float)
        self.child_number_visits = defaultdict(float)


def UCT_search(state, num_reads):
    root = MCTSNode(state, move=None, parent=SentinelNode())
    for _ in range(num_reads):
        leaf = root.select_leaf()
        child_priors, value_estimate = NeuralNet.evaluate(leaf.state)
        leaf.expand(child_priors)
        leaf.back_update(value_estimate)
    return np.argmax(root.child_number_visits)


class NeuralNet:
    @classmethod
    def evaluate(self, game_state):
        return np.random.random([BOARD_SIZE_LEN+1]), np.random.random()


class GameState:
    def __init__(self, to_play=1):
        self.to_play = to_play

    def take_move(self, move):
        return GameState(-self.to_play)


if __name__ == '__main__':
    num_reads = 10000
    import time
    tick = time.time()
    UCT_search(GameState(), num_reads)
    tock = time.time()
    print("Took %s sec to run %s times" % (tock - tick, num_reads))
    import resource
    print("Consumed %sB memory" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)