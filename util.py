class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def flatten(t):
    return [item for sublist in t for item in sublist]


def int_move_to_position(int_move):
    if int_move == 64:
        return "pass"
    line = int_move // 8
    col = int_move % 8
    line += 1
    col = chr(ord('A') + col)
    return "{}{}".format(line, col)


def position_to_int_move(pos):
    if pos == 'pass':
        return 64
    line = pos[0]
    col = pos[1]
    line = int(line) - 1
    col = ord(col) - ord('A')
    return 8*line+col
