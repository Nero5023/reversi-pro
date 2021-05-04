from mcts import MCTS
import config
from board import GameState
from nnet import NeuralNet
from util import position_to_int_move, int_move_to_position
import numpy as np
from subprocess import PIPE, STDOUT, Popen


class GamePlayer:
    def take_move(self, move):
        pass

    def pick_move(self, game_state: GameState):
        pass

    def rival_take_move(self, move):
        pass


class EdaxPlayer(GamePlayer):
    def __init__(self, level):
        edax_exec = config.edax_path + " -q -eval-file " + config.edax_eval_path \
                    + " -book-file " + config.edax_book_path + " --level " + str(level) + " -book-randomness 10"
        self.edax = Popen(edax_exec, shell=True, stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        self.read_stdout()

    def take_move(self, move):
        pass

    def pick_move(self, game_state: GameState):
        self.write_stdin("go")
        edax_move = self.read_stdout().split("plays ")[-1][:2]
        if edax_move == "PS":
            return config.pass_move
        else:
            edax_move = edax_move[::-1]
            if edax_move.strip() == '':
                return config.pass_move
            return position_to_int_move(edax_move)

    def rival_take_move(self, move):
        if move == config.pass_move:
            self.write_stdin("pass")
        else:
            ch_move = int_move_to_position(move)
            # reverse it to meet edax protocal
            ch_move = ch_move[::-1]
            self.write_stdin(ch_move)
        self.read_stdout()

    def write_stdin(self, command):
        self.edax.stdin.write(str.encode(command + "\n"))
        self.edax.stdin.flush()

    def read_stdout(self):
        out = b''
        while True:
            next_b = self.edax.stdout.read(1)
            if next_b == b'>' and ((len(out) > 0 and out[-1] == 10) or len(out) == 0):
                break
            else:
                out += next_b
        return out.decode("utf-8")

    def close(self):
        self.edax.terminate()


class MCTSPlayer(GamePlayer):
    def __init__(self, nn: NeuralNet, sim_num=config.play_simu_num, print_value=True):
        self.mcts = MCTS(nn)
        self.nn = nn
        self.sum_num = sim_num
        self.print_value = print_value

    def pick_move(self, game_state):
        if self.print_value:
            _, value = self.nn.predict(self.mcts.current_node.to_features())
            print("current state predict value: {}".format(value))
        self.mcts.search(self.sum_num)
        move = self.mcts.pick_move()
        if self.print_value:
            print("After search Q value: {}".format(self.mcts.current_node.Q))
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
            legal_moves = np.where(game_state.get_legal_actions() == 1)[0]
            for i in legal_moves:
                print(int_move_to_position(i))
            human_input_str = input(">")
            human_input_str = human_input_str.upper()
            if human_input_str == "PASS":
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
            if move not in legal_moves:
                print("not legal move!")
                continue
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
            print("Black: move {}: {}".format(move_black, int_move_to_position(move_black)))
            print(game.board.to_str())

        if game.is_terminal:
            break

        player_white.rival_take_move(move_black)

        move_white = player_white.pick_move(game)
        player_white.take_move(move_white)

        game = game.take_move(move_white)

        if need_print:
            print("White: move {}: {}".format(move_white, int_move_to_position(move_white)))
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


def play_model_reversi_with_edax(model_path, level, times=20):
    black_win = 0
    tie = 0
    white_win = 0
    for i in range(times):
        nn0 = NeuralNet(config.game_config)
        nn0.load_checkpoint(filename=model_path)
        player0 = MCTSPlayer(nn0)

        player_edax = EdaxPlayer(level)
        res = play_reversi(player0, player_edax)
        print(res)
        if res == 1:
            black_win += 1
        elif res == 0:
            tie += 1
        else:
            white_win += 1
    print("win: {}, tie: {}, white: {}".format(black_win, tie, white_win))
    print("percentage: {}".format(black_win/times))


def play_model_with_human(model_path):
    nn0 = NeuralNet(config.game_config)
    nn0.load_checkpoint(filename=model_path)
    player0 = MCTSPlayer(nn0)
    human = HumanPlayer()
    play_reversi(player0, human)


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

    # nn0 = NeuralNet(config.game_config)
    # nn0.load_checkpoint(filename='model_v.tar')
    # player0 = MCTSPlayer(nn0)
    # human = HumanPlayer()
    # play_reversi(player0, human)

    # play_model_with_human('model_v.tar')

    play_model_reversi_with_edax('model_v.tar', level=2, times=10)