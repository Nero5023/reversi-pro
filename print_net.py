# from .config import game_config
import config
from nnet import NNetModel
import sys
import argparse


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-t', '--type', type=int, dest="type", default=1, required=False)
    return args.parse_args()

if __name__ == '__main__':
    from torchsummary import summary
    net = NNetModel(config.game_config)
    summary(net, (7,8,8), batch_size=-1)
    print(parse_args().type)