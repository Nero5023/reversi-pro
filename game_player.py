from mcts import MCTS
import config
from board import GameState
from nnet import NeuralNet


class GamePlayer:
    def take_move(self, move):
        pass

    def pick_move(self):
        pass

    def rival_take_move(self, move):
        pass


class MCTSPlayer(GamePlayer):
    def __init__(self, nn, sim_num=config.play_simu_num):
        self.mcts = MCTS(nn)
        self.sum_num = sim_num

    def pick_move(self):
        self.mcts.search(self.sum_num)
        move = self.mcts.pick_move()
        return move

    def take_move(self, move):
        self.mcts.take_move(move)

    def rival_take_move(self, move):
        self.mcts.take_move(move)


def play_reversi(player_black: GamePlayer, player_white: GamePlayer, need_print=True):
    game = GameState.INIT_State()
    if need_print:
        print(game.board.to_str())
        print("start")
    while not game.is_terminal:
        move_black = player_black.pick_move()
        player_black.take_move(move_black)

        game = game.take_move(move_black)

        if need_print:
            print("Black: move {}".format(move_black))
            print(game.board.to_str())
        player_white.rival_take_move(move_black)
        move_white = player_white.pick_move()
        player_white.take_move(move_white)

        game = game.take_move(move_white)

        if need_print:
            print("White: move {}".format(move_white))
            print(game.board.to_str())

        player_black.rival_take_move(move_white)

    # return 1 if black win elif white win return -1 else tie: return 0
    return game.winner_score()


if __name__ == '__main__':
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    nn0 = NeuralNet(config.game_config)
    nn0.load_checkpoint('best_model.pth.tar')
    nn1 = NeuralNet(config.game_config)

    player0 = MCTSPlayer(nn0)
    player1 = MCTSPlayer(nn1)

    res = play_reversi(player0, player1)
    print(res)