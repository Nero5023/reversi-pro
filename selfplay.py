from nnet import NeuralNet
from util import dotdict
from board import BOARD_SIDE
from mcts import TOTAL_POSSIBLE_MOVE, MCTS, NeuralNetRandom
import config

# Handle problem OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

game_config = dotdict({
    "board_size": (BOARD_SIDE, BOARD_SIDE),
    "action_size":  TOTAL_POSSIBLE_MOVE,
    "feature_channels": 7
})


class SelfPlay:
    def __init__(self, epoch_max=config.self_play_epoch_max, simu_num=config.simu_num):
        # self.nn = NeuralNet(game_config)
        self.nn = NeuralNetRandom()
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
        iter = 0
        while not mcts.is_terminal:
            mcts.search(self.simu_num)
            move = mcts.pick_move()
            mcts.take_move(move)
            print("Move iter: {}".format(iter))
            iter += 1
        return mcts


if __name__ == '__main__':
    s = SelfPlay()
    s.start()


