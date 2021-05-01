import sys

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from . import net_config


class NNetModelV2(nn.Module):
    def __init__(self, game_config):
        # game params
        self.board_x, self.board_y = game_config.board_size
        self.action_size = game_config.action_size

        super(NNetModelV2, self).__init__()
        self.conv1 = nn.Conv2d(game_config.feature_channels_v2, 64, 3, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256*(self.board_x+2)*(self.board_y+2), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        # s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = F.relu(self.bn5(self.conv5(s)))
        s = F.relu(self.bn6(self.conv6(s)))
        s = s.view(-1, 256*(self.board_x+2)*(self.board_y+2))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=net_config.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=net_config.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)


# if __name__ == '__main__':
#     class dotdict(dict):
#         """dot.notation access to dictionary attributes"""
#         __getattr__ = dict.get
#         __setattr__ = dict.__setitem__
#         __delattr__ = dict.__delitem__
#
#     game_config = dotdict({
#         "board_size": (8, 8),
#         "action_size":  8*8+1,
#         "feature_channels": 7,
#         "feature_channels_v2": 15,
#     })
#     from torchsummary import summary
#     net = NNetModelV2(game_config)
#     summary(net, (15,8,8), batch_size=-1)
#     # 加上边的 feature 和加上角的 feature, 加上稳定子