import torch

batch_size = 32
cuda = torch.cuda.is_available()
num_channels = 512
lr = 0.001
epochs = 50
dropout = 0.3
l2_constant = 1e-4

log_path = 'log'
