# from .config import game_config
import config
from nnet import NNetModel, NNetModelV2
import sys
import argparse


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-t', '--type', type=int, dest="type", default=1, required=False)
    return args.parse_args()

if __name__ == '__main__':
    # from torchsummary import summary
    # net = NNetModel(config.game_config)
    # summary(net, (7,8,8), batch_size=-1)
    # print(parse_args().type)

    # model v1
    import netron
    import torch
    import torch.onnx
    net = NNetModel(config.game_config)
    d = torch.rand(1, 7, 8, 8)
    onnx_path = "model_v1.onnx"
    torch.onnx.export(net, d, onnx_path)

    net2 = NNetModelV2(config.game_config)
    d = torch.rand(1, 15, 8, 8)
    onnx_path = "model_v2.onnx"
    torch.onnx.export(net2, d, onnx_path)
