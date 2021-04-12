from enum import Enum
import numpy as np


class Player(Enum):
    BLACK = 0
    WHITE = 1

    def rival(self):
        if self == Player.BLACK:
            return Player.WHITE
        else:
            return Player.BLACK


BLACK_START = np.uint64(0b1000 << 32 | 0b10000 << 24)
WHITE_START = np.uint64(0b1000 << 24 | 0b10000 << 32)

BOARD_SIDE = 8
BOARD_SIZE_LEN = BOARD_SIDE*BOARD_SIDE
PASS_MOVE = BOARD_SIZE_LEN


class ReversiBoard:
    def __init__(self, black_state=BLACK_START, white_state=WHITE_START):
        self.black_bit = black_state
        self.white_bit = white_state
        self.black_2d = bit_to_2d_array(self.black_bit, BOARD_SIDE, BOARD_SIDE)
        self.white_2d = bit_to_2d_array(self.white_bit, BOARD_SIDE, BOARD_SIDE)

    def bit_state(self, player: Player):
        if player == Player.BLACK:
            return self.black_bit
        else:
            return self.white_bit

    def array2d_state(self, player: Player):
        if player == Player.BLACK:
            return self.black_2d
        else:
            return self.white_2d

    def get_self_rival_bit_tuple(self, player: Player):
        return self.bit_state(player), self.bit_state(player.rival())

    def get_self_rival_array2d_tuple(self, player: Player):
        return self.array2d_state(player), self.array2d_state(player.rival())

    def get_legal_actions(self, player: Player):
        self_s, rival_s = self.get_self_rival_bit_tuple(player)
        legal_moves = bit_to_1d_array(get_legal_moves_bit(self_s, rival_s), BOARD_SIZE_LEN)
        return legal_moves

    def get_legal_actions(self, player: Player):
        self_s, rival_s = self.get_self_rival_bit_tuple(player)
        legal_moves = bit_to_1d_array(get_legal_moves_bit(self_s, rival_s), BOARD_SIZE_LEN)
        return legal_moves

    def get_legal_actions_in_numbers(self, player: Player):
        actions = self.get_legal_actions(player)
        return np.where(actions == 1)

    def task_move(self, player: Player, move):
        if move == PASS_MOVE:
            return ReversiBoard(self.black_bit, self.white_bit)
        bit_move = np.uint64(0b1 << move)
        self_s, rival_s = self.get_self_rival_bit_tuple(player)
        flipped_stones = get_flipped_stones_bit(bit_move, self_s, rival_s)
        self_s |= flipped_stones | bit_move
        rival_s &= ~flipped_stones
        if player == Player.BLACK:
            return ReversiBoard(self_s, rival_s)
        else:
            return ReversiBoard(rival_s, self_s)

    def to_str(self, player: Player = None):
        first_row = '  A B C D E F G H'
        if player is None:
            zip_list = zip(bit_to_1d_array(self.black_bit, BOARD_SIZE_LEN),
                           bit_to_1d_array(self.white_bit, BOARD_SIZE_LEN))
        else:
            zip_list = zip(bit_to_1d_array(self.black_bit, BOARD_SIZE_LEN),
                           bit_to_1d_array(self.white_bit, BOARD_SIZE_LEN),
                           self.get_legal_actions(player))

        # print(zip_list)
        board_ch_list = np.array(list(map(map_tuple_to_ch, zip_list))).reshape(BOARD_SIDE, BOARD_SIDE)
        rep_str_arr = [first_row]
        for index, arr in enumerate(board_ch_list):
            row = '{} {}'.format(index+1, ' '.join(arr))
            rep_str_arr.append(row)
        return '\n'.join(rep_str_arr)


def map_tuple_to_ch(tup):
    black = '●'
    white = '○'
    legal = '×'
    empty = '☐'
    if len(tup) == 2:
        if tup == (1, 0):
            return black
        elif tup == (0, 1):
            return white
        else:
            return empty
    elif len(tup) == 3:
        if tup == (1, 0, 0):
            return black
        elif tup == (0, 1, 0):
            return white
        elif tup == (0, 0, 1):
            return legal
        else:
            return empty

# TODO: Change


left_right_mask = np.uint64(0x7e7e7e7e7e7e7e7e)
top_bottom_mask = np.uint64(0x00ffffffffffff00)
corner_mask = left_right_mask & top_bottom_mask


def bit_to_1d_array(bit, size):
    return np.array(list(reversed((("0" * size) + bin(bit)[2:])[-size:])), dtype=np.uint8)


def bit_to_2d_array(bit, h, w):
    return bit_to_1d_array(bit, h*w).reshape(h, w)


def get_legal_moves_bit(own, enemy):
    legal_moves = np.uint64(0)
    legal_moves |= search_legal_moves_left(own, enemy, left_right_mask, np.uint64(1))
    legal_moves |= search_legal_moves_left(own, enemy, corner_mask, np.uint64(9))
    legal_moves |= search_legal_moves_left(own, enemy, top_bottom_mask, np.uint64(8))
    legal_moves |= search_legal_moves_left(own, enemy, corner_mask, np.uint64(7))
    legal_moves |= search_legal_moves_right(own, enemy, left_right_mask, np.uint64(1))
    legal_moves |= search_legal_moves_right(own, enemy, corner_mask, np.uint64(9))
    legal_moves |= search_legal_moves_right(own, enemy, top_bottom_mask, np.uint64(8))
    legal_moves |= search_legal_moves_right(own, enemy, corner_mask, np.uint64(7))
    legal_moves &= ~(own | enemy)
    return legal_moves


def search_legal_moves_left(own, enemy, mask, offset):
    return search_contiguous_stones_left(own, enemy, mask, offset) >> offset


def search_legal_moves_right(own, enemy, mask, offset):
    return search_contiguous_stones_right(own, enemy, mask, offset) << offset


def get_flipped_stones_bit(bit_move, own, enemy):
    flipped_stones = np.uint64(0)
    flipped_stones |= search_flipped_stones_left(bit_move, own, enemy, left_right_mask, np.uint64(1))
    flipped_stones |= search_flipped_stones_left(bit_move, own, enemy, corner_mask, np.uint64(9))
    flipped_stones |= search_flipped_stones_left(bit_move, own, enemy, top_bottom_mask, np.uint64(8))
    flipped_stones |= search_flipped_stones_left(bit_move, own, enemy, corner_mask, np.uint64(7))
    flipped_stones |= search_flipped_stones_right(bit_move, own, enemy, left_right_mask, np.uint64(1))
    flipped_stones |= search_flipped_stones_right(bit_move, own, enemy, corner_mask, np.uint64(9))
    flipped_stones |= search_flipped_stones_right(bit_move, own, enemy, top_bottom_mask, np.uint64(8))
    flipped_stones |= search_flipped_stones_right(bit_move, own, enemy, corner_mask, np.uint64(7))
    return flipped_stones


def search_flipped_stones_left(bit_move, own, enemy, mask, offset):
    flipped_stones = search_contiguous_stones_left(bit_move, enemy, mask, offset)
    if own & (flipped_stones >> offset) == np.uint64(0):
        return np.uint64(0)
    else:
        return flipped_stones


def search_flipped_stones_right(bit_move, own, enemy, mask, offset):
    flipped_stones = search_contiguous_stones_right(bit_move, enemy, mask, offset)
    if own & (flipped_stones << offset) == np.uint64(0):
        return np.uint64(0)
    else:
        return flipped_stones


def search_contiguous_stones_left(own, enemy, mask, offset):
    e = enemy & mask
    s = e & (own >> offset)
    s |= e & (s >> offset)
    s |= e & (s >> offset)
    s |= e & (s >> offset)
    s |= e & (s >> offset)
    s |= e & (s >> offset)
    return s


def search_contiguous_stones_right(own, enemy, mask, offset):
    e = enemy & mask
    s = e & (own << offset)
    s |= e & (s << offset)
    s |= e & (s << offset)
    s |= e & (s << offset)
    s |= e & (s << offset)
    s |= e & (s << offset)
    return s


def bit_count(bit):
    return bin(bit).count('1')


if __name__ == '__main__':
    b = ReversiBoard()
    print(b.get_legal_actions(Player.BLACK))
    print(b.get_legal_actions_in_numbers(Player.BLACK))
    print(b.to_str(Player.BLACK))
    print(b.task_move(Player.BLACK, 44).to_str(Player.WHITE))
