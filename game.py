from .board import Board, PieceEnum


class Game:
    def __init__(self):
        self.board = Board(width=8, height=8)
