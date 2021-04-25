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
C_PUCT = 2
NOISE_EPSILON = 0.25
NOISE_ALPHA = 0.17
FEATURE_NUM = 7


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


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
        self.height = 0
        if parent:
            self.height = parent.height + 1

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
        p_with_noise = self.child_priors*(1-NOISE_EPSILON) + noise*NOISE_EPSILON
        return C_PUCT*math.sqrt(self.N) * (p_with_noise / (1 + self.child_number_visits))

    def best_child(self):
        # if self.is_search_root:
        #     # for search root add noise
        #     return np.argmax(self.child_Q() + self.child_U_inject_noise() + 1000 * self.state.get_legal_actions())
        # else:
        #     # add this to prevent self.child_Q() + self.child_U() < 0, others is == 0, which cloud take illegal action
        #     return np.argmax(self.child_Q() + self.child_U() + 1000 * self.state.get_legal_actions())

        if self.height < 10:
            # for search root add noise
            return np.argmax(self.child_Q() + self.child_U_inject_noise() + 1000 * self.state.get_legal_actions())
        else:
            # add this to prevent self.child_Q() + self.child_U() < 0, others is == 0, which cloud take illegal action
            return np.argmax(self.child_Q() + self.child_U() + 1000 * self.state.get_legal_actions())

        # TODO: where to add noise
        # return np.argmax(self.child_Q() + self.child_U() + 1000 * self.state.get_legal_actions())

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
        if np.sum(self.child_number_visits) == 0 or self.state.need_pass():
            # TODO: if this return pass move
            self.pi = np.zeros([TOTAL_POSSIBLE_MOVE], dtype=np.float)
            self.pi[PASS_MOVE] = 1
        else:
            # probs = self.child_number_visits ** (1 / temperature)
            # sum_probs = np.sum(probs)
            # self.pi = probs/sum_probs
            # TODO: check if this correct
            pi = softmax(1.0/temperature * np.log(self.child_number_visits + 1e-10))
            self.pi = pi
        return self.pi

    # TODO: add noise

    def generate_flip_rotate_data(self, winner_z):
        """
        generate the data set by rotation and flipping
        :param state: np array: feature_nums x board_size x board_size
        :param pi:
        :param winner_z: 1: black win   -1: white win
        :return: extended features [(state_features, pi, winner_z),...]
        """
        features = self.to_features()
        pi = self.pi
        extended = []
        for i in [0, 1, 2, 3]:
            # rotate
            new_state = np.array([np.rot90(s, i) for s in features])
            pi_without_pass = pi[:-1].reshape(BOARD_SIDE, BOARD_SIDE)
            new_pi = np.rot90(pi_without_pass, i)
            extended.append(
                (
                    new_state,
                    np.append(new_pi.flatten(), pi[-1]),
                    winner_z
                )
            )
            # flip left right(mirror)
            new_state_mirror = np.array([np.fliplr(s) for s in new_state])
            new_pi_mirror = np.fliplr(new_pi)
            extended.append(
                (
                    new_state_mirror,
                    np.append(new_pi_mirror.flatten(), pi[-1]),
                    winner_z
                )
            )
        return extended


class SentinelNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_values = defaultdict(float)
        self.child_number_visits = defaultdict(float)
        self.height = -1


def UCT_search(state, num_reads):
    root = MCTSNode(state, move=None, parent=SentinelNode())
    for _ in range(num_reads):
        leaf = root.select_leaf()
        child_priors, value_estimate = NeuralNetRandom.evaluate(leaf.state)
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
            child_priors, value_estimate = self.nn.predict(leaf.to_features())
            # TODO: mask probs?
            leaf.expand(child_priors)
            leaf.back_update(value_estimate)

    def take_move(self, move):
        # pi = self.current_node.children_pi(self.temperature)
        # move = pi.argmax()
        self.current_node = self.current_node.maybe_add_child(move)
        self.current_node.is_search_root = True
        self.move_num += 1

        if self.current_node.is_terminal:
            # update last node of children's pi
            self.current_node.children_pi(self.temperature)
            print("Termail")
            print(self.current_node.state.board.to_str())
            print("WINNER: {}".format(self.current_node.state.winner()))
            self.winner = self.current_node.state.winner()

    def pick_move(self):
        pi = self.current_node.children_pi(self.temperature)
        move = pi.argmax()
        return move

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

    @property
    def is_terminal(self):
        return self.current_node.is_terminal

    def generate_game_data(self):
        if not self.is_terminal:
            return []
        winner_z = 0
        if self.winner == Player.BLACK:
            winner_z = 1
        elif self.winner == Player.WHITE:
            winner_z = -1
        node = self.current_node
        data = []
        while True:
            # extend_datas = generate_flip_rotate_data(node.to_features(), node.pi, winner_z)
            extend_datas = node.generate_flip_rotate_data(winner_z)
            data.extend(extend_datas)
            if node.is_game_root:
                break
            node = node.parent
        return data


def temperature_func(move_num):
    if move_num <= 10:
        return 1
    else:
        return 0.95**(move_num - 10)


class MCTSBatch:
    def __init__(self, nn, batch_size: int):
        self.nn = nn
        self.roots = []
        self.current_nodes = []
        self.batch_size = batch_size
        self.terminal_count = 0
        self.winners = [0]*batch_size
        for i in range(batch_size):
            sentinel_node = SentinelNode()
            root = MCTSNode(GameState.INIT_State(), PASS_MOVE, sentinel_node)
            root.is_game_root = True
            root.is_search_root = True
            self.roots.append(root)
            self.current_nodes.append(root)

    def search(self, num_sims):
        for _ in range(num_sims):
            # terminal_leaves = []
            # TODO: or make all leaves to predict
            non_terminal_leaves = []
            for current_node in self.current_nodes:
                leaf = current_node.select_leaf()
                if leaf.is_terminal:
                    leaf.back_update(leaf.state.winner_score())
                    continue
                non_terminal_leaves.append(leaf)
            if len(non_terminal_leaves) == 0:
                continue
            batch_features = np.zeros([len(non_terminal_leaves), FEATURE_NUM, BOARD_SIDE, BOARD_SIDE], dtype=np.float)
            for i, nt_leaf in enumerate(non_terminal_leaves):
                batch_features[i] = nt_leaf.to_features()
            child_priors_batch, value_estimate_batch = self.nn.predict_batch(batch_features)
            for i, nt_leaf in enumerate(non_terminal_leaves):
                nt_leaf.expand(child_priors_batch[i])
                nt_leaf.back_update(value_estimate_batch[i])

    def pick_moves(self):
        return [node.children_pi(temperature_func(node.height)).argmax() for node in self.current_nodes]

    def take_moves(self, moves):
        assert len(moves) == len(self.current_nodes)
        for i in range(len(self.current_nodes)):
            node = self.current_nodes[i]
            if node.is_terminal:
                continue
            move = moves[i]
            self.current_nodes[i] = node.maybe_add_child(move)
            self.current_nodes[i].is_search_root = True

            if self.current_nodes[i].is_terminal:
                # update last node of children's pi
                self.current_nodes[i].children_pi(temperature_func(self.current_nodes[i].height))
                self.terminal_count += 1
                self.winners[i] = self.current_nodes[i].state.winner()

    def search_and_pick_to_move(self, num_sims):
        self.search(num_sims)
        moves = self.pick_moves()
        self.take_moves(moves)

    @property
    def all_terminal(self):
        return self.terminal_count == self.batch_size

    def generate_game_data(self):
        if not self.all_terminal:
            return []
        data = []
        for i in range(self.batch_size):
            winner_z = 0
            if self.winners[i] == Player.BLACK:
                winner_z = 1
            elif self.winners[i] == Player.WHITE:
                winner_z = -1
            node = self.current_nodes[i]
            while True:
                # extend_datas = generate_flip_rotate_data(node.to_features(), node.pi, winner_z)
                extend_datas = node.generate_flip_rotate_data(winner_z)
                data.extend(extend_datas)
                if node.is_game_root:
                    break
                node = node.parent
        return data


class NeuralNetRandom:
    def predict(self, features):
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

    mcts = MCTS(NeuralNetRandom())
    move_num = 1
    while True:
        print("Move id {}".format(move_num))
        mcts.search(num_reads)
        print(mcts.current_node.state.board.to_str())
        move = mcts.pick_move()
        mcts.take_move(move)
        move_num += 1
