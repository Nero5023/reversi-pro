from .board import *


# class Game:
#     def __init__(self):
#         self.board = ReversiBoard(width=8, height=8)
#
# class GameState:
#     def __init__(self, board: ReversiBoard, to_play: Player):
#         self.board = board
#         self.to_play = to_play
#
#     @property
#     def is_terminal(self):
#         return (self.board.black_bit | self.board.white_bit) == ~np.uint64(0)
#
#     def take_move(self):
#