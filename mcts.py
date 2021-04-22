import math

# W: total weight
# N: number visit
#

from board import BOARD_SIZE_LEN, PASS_MOVE, ReversiBoard, GameState, BOARD_SIDE, Player
import numpy as np
from collections import defaultdict

TOTAL_POSSIBLE_MOVE = BOARD_SIZE_LEN + 1

# TODO: add config file

# c_puct
# find a best C_PUCT
C_PUCT = 1
NOISE_EPSILON = 0.25
NOISE_ALPHA = 0.17

class MCTSNode:
    def __init__(self, state: GameState, move, parent=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.is_expanded = False
        self.is_game_root = False
        self.is_search_root = False
        self.__is_terminal = False
        self.children = {}
        self.parent = parent
        self.pi = np.zeros([TOTAL_POSSIBLE_MOVE], dtype=np.float32)
        # +1 for pass move
        self.child_priors = np.zeros([TOTAL_POSSIBLE_MOVE], dtype=np.float32)
        self.child_total_values = np.zeros([TOTAL_POSSIBLE_MOVE], dtype=np.float32)
        self.child_number_visits = np.zeros([TOTAL_POSSIBLE_MOVE], dtype=np.float32)

    @property
    def is_terminal(self):
        if self.__is_terminal is True:
            return self.__is_terminal
        # TODO: add is_terminal API
        self.__is_terminal = self.state.is_terminal
        return self.__is_terminal

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
        return self.child_total_values * self.state.to_play_factor / (self.child_number_visits + (self.child_number_visits == 0))

    def child_U(self):
        # self.edge_P * math.sqrt(max(1, self.self_N)) / (1 + self.edge_N)
        return C_PUCT*math.sqrt(self.N) * (self.child_priors / (1 + self.child_number_visits))

    def child_U_inject_noise(self):
        epsilon = 1e-5
        legal_moves = self.state.get_legal_actions() + epsilon
        alphas = legal_moves * ([NOISE_ALPHA] * TOTAL_POSSIBLE_MOVE)
        noise = np.random.dirichlet(alphas)
        p_with_noise = self.child_priors*(1-NOISE_EPSILON) + noise + NOISE_EPSILON
        return C_PUCT*math.sqrt(self.N) * (p_with_noise / (1 + self.child_number_visits))

    def best_child(self):
        # if self.is_search_root:
        #     # for search root add noise
        #     return np.argmax(self.child_Q() + self.child_U_inject_noise() + 1000 * self.state.get_legal_actions())
        # else:
        #     # add this to prevent self.child_Q() + self.child_U() < 0, others is == 0, which cloud take illegal action
        #     return np.argmax(self.child_Q() + self.child_U() + 1000 * self.state.get_legal_actions())

        # TODO: where to add noise
        return np.argmax(self.child_Q() + self.child_U() + 1000 * self.state.get_legal_actions())

    def select_leaf(self):
        node = self
        while node.is_expanded:
            if node.is_terminal:
                break
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
        self.is_expanded = True
        # normalize
        priors = np.multiply(child_prior_probabilities, self.state.get_legal_actions())
        normalized = priors/np.sum(priors)

        self.child_priors = normalized

    def to_features(self):
        history_num = 2
        num_of_features = 7
        player = self.state.to_play
        features = np.zeros([7, BOARD_SIDE, BOARD_SIDE], dtype=np.float)
        node = self
        for i in range(history_num):
            me, rival = node.state.board.get_self_rival_array2d_tuple(player)
            features[2*i] = me
            features[2*i+1] = rival
            if node.is_game_root:
                break
            node = node.parent
        me_l, rival_l = self.state.board.get_self_rival_legal_action_2d_tuple(player)
        features[history_num*2] = me_l
        features[history_num * 2 + 1] = rival_l
        if player == Player.BLACK:
            features[-1] = np.ones([BOARD_SIDE, BOARD_SIDE], dtype=np.float)
        return features

    def back_update(self, value):
        # TODO: Check minigo mcts.py line 210
        # value = 1, black win
        # value = -1, white win
        node = self
        # check if node is root
        while True:
            node.N += 1
            node.W += value

            if node.is_search_root:
                break
            node = node.parent

    # FOR bias
    # TODO: add noise
    def children_pi(self, temperature):
        # todo: check possible move
        # TODO: maybe overflow here
        # /Users/Nero/local_dev/nyu/ml/ml-proj/mcts.py:104: RuntimeWarning: overflow encountered in power
        #   probs = self.child_number_visits ** (1 / temperature)
        # /Users/Nero/local_dev/nyu/ml/ml-proj/mcts.py:111: RuntimeWarning: invalid value encountered in true_divide
        #   self.pi = probs/sum_probs
        probs = self.child_number_visits ** (1 / temperature)
        sum_probs = np.sum(probs)
        if sum_probs == 0 or self.state.need_pass():
            # TODO: if this return pass move
            self.pi = np.zeros([TOTAL_POSSIBLE_MOVE], dtype=np.float)
            self.pi[PASS_MOVE] = 1
        else:
            self.pi = probs/sum_probs
        return self.pi

    # TODO: add noise


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


# TODO: virtual loss for parallel tree search
class MCTS:
    def __init__(self, nn):
        self.nn = nn
        sentinel_node = SentinelNode()
        self.root = MCTSNode(GameState.INIT_State(), PASS_MOVE, sentinel_node)
        self.root.is_game_root = True
        self.root.is_search_root = True
        self.current_node = self.root
        self.move_num = 0
        self.winner = None

    def search(self, num_sims):
        for _ in range(num_sims):
            leaf = self.current_node.select_leaf()
            if leaf.is_terminal:
                leaf.back_update(leaf.state.winner_score())
                continue
            child_priors, value_estimate = self.nn.evaluate(leaf.to_features())
            # TODO: mask probs?
            leaf.expand(child_priors)
            leaf.back_update(value_estimate)

    def take_move(self):
        pi = self.current_node.children_pi(self.temperature)
        move = pi.argmax()
        self.current_node = self.current_node.maybe_add_child(move)
        self.current_node.is_search_root = True

        if self.current_node.is_terminal:
            print("Termail")
            print(self.current_node.state.board.to_str())
            print("WINNER: {}".format(self.current_node.state.winner()))
            self.winner = self.current_node.state.winner()
            assert False
        self.move_num += 1

    def normalize_with_legal_moves(self, child_priors, legal_moves):
        legal_probs = np.multiply(child_priors, legal_moves)
        return legal_probs/np.sum(legal_probs)


    # TODO: check temperature strategy
    @property
    def temperature(self):
        if self.move_num <= 10:
            return 1
        else:
            return 0.95**(self.move_num - 10)


class NeuralNet:
    def evaluate(self, features):
        value = np.random.random()*2 -1
        return np.random.random([TOTAL_POSSIBLE_MOVE]), value


# class GameState:
#     def __init__(self, to_play=1):
#         self.to_play = to_play
#
#     def take_move(self, move):
#         return GameState(-self.to_play)


if __name__ == '__main__':
    num_reads = 500
    # import time
    # tick = time.time()
    # UCT_search(GameState(), num_reads)
    # tock = time.time()
    # print("Took %s sec to run %s times" % (tock - tick, num_reads))
    # import resource
    # print("Consumed %sB memory" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    mcts = MCTS(NeuralNet())
    move_num = 1
    while True:
        print("Move id {}".format(move_num))
        mcts.search(num_reads)
        print(mcts.current_node.state.board.to_str())
        mcts.take_move()
        move_num+=1
