import torch
from torchvision.transforms import Lambda

RGBToBGR = Lambda(lambda x: torch.stack([x[2], x[1], x[0]])) # x:torch.Tensor
