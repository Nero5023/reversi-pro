import torch

batch_size = 64
cuda = torch.cuda.is_available()
num_channels = 512
lr = 0.001
epochs = 10
dropout = 0.3
l2_constant = 1e-4

