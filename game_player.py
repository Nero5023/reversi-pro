from mcts import MCTS
import config
from board import GameState
from nnet import NeuralNet
from util import position_to_int_move, int_move_to_position
import numpy as np


class GamePlayer:
    def take_move(self, move):
        pass

    def pick_move(self, game_state: GameState):
        pass

    def rival_take_move(self, move):
        pass


class MCTSPlayer(GamePlayer):
    def __init__(self, nn, sim_num=config.play_simu_num):
        self.mcts = MCTS(nn)
        self.sum_num = sim_num

    def pick_move(self, game_state):
        self.mcts.search(self.sum_num)
        move = self.mcts.pick_move()
        return move

    def take_move(self, move):
        self.mcts.take_move(move)

    def rival_take_move(self, move):
        self.mcts.take_move(move)


class HumanPlayer(GamePlayer):
    def pick_move(self, game_state: GameState):
        while True:
            print("Please input:")
            print("legal action:")
            for i in np.where(game_state.get_legal_actions() == 1)[0]:
                print(int_move_to_position(i))
            human_input_str = input(">")
            if human_input_str == "pass":
                return 64
            if len(human_input_str) != 2:
                print("input is two char")
                continue
            if not human_input_str[0].isnumeric():
                continue
            row = int(human_input_str[0])
            if row > 8 or row < 0:
                continue
            col = human_input_str[1]
            if ord(col) < ord('A') or ord(col) > ord('H'):
                continue
            move = position_to_int_move(human_input_str)
            return move


def play_reversi(player_black: GamePlayer, player_white: GamePlayer, need_print=True):
    game = GameState.INIT_State()
    if need_print:
        print(game.board.to_str())
        print("start")
    while not game.is_terminal:
        move_black = player_black.pick_move(game)
        player_black.take_move(move_black)

        game = game.take_move(move_black)

        if need_print:
            print("Black: move {}".format(move_black))
            print(game.board.to_str())
        player_white.rival_take_move(move_black)
        move_white = player_white.pick_move(game)
        player_white.take_move(move_white)

        game = game.take_move(move_white)

        if need_print:
            print("White: move {}".format(move_white))
            print(game.board.to_str())

        player_black.rival_take_move(move_white)

    # return 1 if black win elif white win return -1 else tie: return 0
    return game.winner_score()


def play_reversi_benchmark(model_path0, model_path1, nums):
    black_win = 0
    tie = 0
    white_win = 0
    for i in range(nums):
        nn0 = NeuralNet(config.game_config)
        nn0.load_checkpoint(filename=model_path0)
        nn1 = NeuralNet(config.game_config)
        nn1.load_checkpoint(filename=model_path1)
        player0 = MCTSPlayer(nn0)
        player1 = MCTSPlayer(nn1)
        res = play_reversi(player0, player1)
        print(res)
        if res == 1:
            black_win += 1
        elif res == 0:
            tie += 1
        else:
            white_win += 1
    print("win: {}, tie: {}, white: {}".format(black_win, tie, white_win))
    print("percentage: {}".format(black_win/nums))


if __name__ == '__main__':
    # import os
    #
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    #
    #
    #
    # play_reversi_benchmark('model_v36.tar', 'model_v10.tar', 10)

    # human_input_str = input(">")
    # print(human_input_str)
    nn0 = NeuralNet(config.game_config)
    nn0.load_checkpoint(filename='model_v.tar')
    player0 = MCTSPlayer(nn0)
    human = HumanPlayer()
    play_reversi(human, player0)