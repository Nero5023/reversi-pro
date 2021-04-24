import os
import sys
import time

import numpy as np
from tqdm import tqdm


import torch
import torch.optim as optim

from .nn_model import NNetModel
from . import net_config


class AverageMeter(object):

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class NeuralNet:
    def __init__(self, game_config):
        self.nnet = NNetModel(game_config)
        self.board_x, self.board_y = game_config.board_size
        self.action_size = game_config.action_size

        if net_config.cuda:
            self.nnet.cuda()

    def train(self, examples, version=0):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        with open(net_config.log_path + '/' + "loss_log.csv", "a+") as loss_log_file:

            optimizer = optim.Adam(self.nnet.parameters(), weight_decay=net_config.l2_constant)

            for epoch in range(net_config.epochs):
                print('EPOCH ::: ' + str(epoch + 1))
                self.nnet.train()
                pi_losses = AverageMeter()
                v_losses = AverageMeter()

                batch_count = int(len(examples) / net_config.batch_size)

                t = tqdm(range(batch_count), desc='Training Net')
                step = 0
                for _ in t:
                    sample_ids = np.random.randint(len(examples), size=net_config.batch_size)
                    boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                    boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                    target_pis = torch.FloatTensor(np.array(pis))
                    target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                    # predict
                    if net_config.cuda:
                        boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                    # compute output
                    out_pi, out_v = self.nnet(boards)
                    l_pi = self.loss_pi(target_pis, out_pi)
                    l_v = self.loss_v(target_vs, out_v)
                    total_loss = l_pi + l_v

                    # log loss
                    if step % 10 == 0:
                        loss_log_file.write('{},{},{},{},{}\n'.format(version, epoch, step, l_pi, l_v))

                    # record loss
                    pi_losses.update(l_pi.item(), boards.size(0))
                    v_losses.update(l_v.item(), boards.size(0))
                    t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    step += 1

    def predict(self, board_features):
        """
        board_features: np array with board features with one board
        """
        # timing
        start = time.time()

        # preparing input
        batch_one = np.array([board_features])
        batch_one = torch.FloatTensor(batch_one.astype(np.float64))
        if net_config.cuda: batch_one = batch_one.contiguous().cuda()
        # board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(batch_one)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def predict_batch(self, batch):
        """
        batch: batch np array with board features: batch_size x features_num x board_size x board_size
        """
        # timing
        start = time.time()

        # preparing input
        batch = torch.FloatTensor(batch.astype(np.float64))
        if net_config.cuda: batch = batch.contiguous().cuda()
        # board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(batch)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy(), v.data.cpu().numpy()

    def loss_pi(self, targets, outputs):
        # return -torch.sum(targets * outputs) / targets.size()[0]
        return -torch.mean(torch.sum(targets * outputs, 1))

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # TODO: maybe save optimizer
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if net_config.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])

