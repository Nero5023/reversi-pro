from nnet import NeuralNet
from util import dotdict
from board import BOARD_SIDE
from mcts import TOTAL_POSSIBLE_MOVE, MCTS, NeuralNetRandom, MCTSBatch
import config
from config import game_config

# Handle problem OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


class SelfPlay:
    def __init__(self, nn, epoch_max=config.self_play_epoch_max, simu_num=config.simu_num):
        # self.nn = NeuralNet(game_config)
        self.nn = nn
        self.epoch_max = epoch_max
        self.simu_num = simu_num
        self.game_data = []

    def start(self):
        for i in range(self.epoch_max):
            print("epoch: {}".format(i))
            mcts = self.play()
            self.game_data.extend(mcts.generate_game_data())

    def play(self):
        mcts = MCTS(self.nn)
        iter = 1
        while not mcts.is_terminal:
            mcts.search(self.simu_num)
            move = mcts.pick_move()
            mcts.take_move(move)
            print("Move iter: {}: {}".format(iter, move))
            iter += 1
        return mcts


class SelfPlayBatch:
    def __init__(self, nn,
                 epoch_max=config.self_play_epoch_max,
                 simu_num=config.simu_num,
                 batch_size=config.self_play_batch_size,
                 generate_feature_version=None):
        self.nn = nn
        self.epoch_max = epoch_max
        self.simu_num = simu_num
        self.game_data = []
        self.batch_size = batch_size
        self.generate_feature_version = generate_feature_version
        if generate_feature_version is None:
            self.generate_feature_version = self.nn.model_type

    def start(self):
        for i in range(self.epoch_max):
            print("epoch: {}".format(i))
            mcts = self.play()
            self.game_data.extend(mcts.generate_game_data(self.generate_feature_version))

    def play(self):
        mcts = MCTSBatch(self.nn, self.batch_size)
        iter = 0
        while not mcts.all_terminal:
            mcts.search_and_pick_to_move(self.simu_num)
            print("Iters: {}".format(iter))
            iter += 1
        return mcts


if __name__ == '__main__':
    s = SelfPlayBatch(NeuralNet(game_config))
    s.start()


