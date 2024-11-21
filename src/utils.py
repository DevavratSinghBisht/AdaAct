import torch.nn as nn

all_acts = [
    nn.ReLU(),
    nn.LeakyReLU(),
    nn.PReLU(),
    nn.ELU(),
    nn.SELU(),
    nn.GELU(),
    nn.Tanh(),
    nn.Sigmoid(),
    nn.Softplus(),
    nn.Softsign(),
    nn.SiLU(),
    nn.Hardswish(),
    nn.Hardtanh(),
    nn.Hardsigmoid(),
    nn.LogSigmoid(),
    nn.Threshold(0.5, 0.0),
    nn.Softmax(dim=-1),
    nn.LogSoftmax(dim=-1)
]