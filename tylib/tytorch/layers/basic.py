import torch
from torch.nn import Module

class Lambda(Module):
    "Create a layer that simply calls `func` with `x`"

    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func=func

    def forward(self, x):
        return self.func(x)