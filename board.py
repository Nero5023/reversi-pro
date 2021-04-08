from enum import Enum


class PieceEnum(Enum):
    BLACK = 1
    WHITE = 2


class Board:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.state = [[0]*self.width for _ in range(self.height)]

    def do_move(self, pos: tuple, piece: PieceEnum):
        x = pos[0]
        y = pos[1]
        self.state[x][y] = piece.value
