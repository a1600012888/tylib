import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import math


class CConv2d(nn.Module):
    def __init__(self, num, in_channels, out_channels, kernel_size, stride=1,
                                 padding=0, dilation=1, groups=1, bias=True):
        super(CConv2d, self).__init__()
        self.num = num
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.routing_fc = nn.Linear(in_channels, num)
        self.weight = Parameter(torch.Tensor(out_channels * in_channels * kernel_size * kernel_size // groups, num))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs):
        '''
        if inputs.shape[-1] == 7:
            print('inputs_se:', inputs_se)
        batchsize, channel, height, width = inputs.shape
        weight = F.linear(inputs_se, self.weight)
        weight = weight.reshape(batchsize * self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size)
        inputs = inputs.reshape(1, batchsize * channel, height, width)
        outputs = F.conv2d(inputs, weight, None, self.stride, self.padding, self.dilation, groups=self.groups * batchsize)
        height, width = outputs.shape[2:]
        outputs = outputs.reshape(batchsize, self.out_channels, height, width)
        if self.bias is not None:
            outputs = outputs + self.bias.reshape(1, -1, 1, 1)
        '''
        x = inputs
        inputs_se = x.reshape(x.shape[0], x.shape[1], -1).mean(dim=-1, keepdim=False)
        inputs_se = F.sigmoid(self.routing_fc(inputs_se))
        # the code below is equal to the above, but faster actually
        #if inputs.shape[-1] == 7:
        #    print('inputs_se:', inputs_se)
        batchsize, channel, height, width = inputs.shape
        weight = self.weight.reshape(self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size, self.num)
        weight = weight.permute(0, 4, 1, 2, 3)
        weight = weight.reshape(self.out_channels * self.num, self.in_channels // self.groups, self.kernel_size, self.kernel_size)
        outputs = F.conv2d(inputs, weight, None, self.stride, self.padding, self.dilation, groups=self.groups)
        outputs = outputs.reshape(outputs.shape[0], outputs.shape[1] // self.num, self.num, outputs.shape[2], outputs.shape[3])
        outputs = outputs * inputs_se.reshape(inputs_se.shape[0], 1, inputs_se.shape[1], 1, 1)
        outputs = outputs.sum(dim=2, keepdim=False)
        #print('aaa')
        if self.bias is not None:
            outputs = outputs + self.bias.reshape(1, -1, 1, 1)
        return outputs

def Conv2Scc(net:nn.Module, num=4):
    import copy
    #dic = copy.deepcopy(dict(net.named_modules()))
    dic = dict(net.named_modules())
    for name, m in dic.items():
        if isinstance(m, nn.Conv2d):
            # print(num, m.in_channels, m.out_channels,
            #                    m.kernel_size, m.stride, m.padding,
            #                    m.dilation, m.groups, m.bias)
            scc_conv = CConv2d(num, m.in_channels, m.out_channels,
                               m.kernel_size[0], m.stride[0], m.padding[0],
                               m.dilation[0], m.groups, m.bias)
            #print('cc')

            names = name.split('.')
            mother = net
            for n in names[:-1]:
                mother = getattr(mother, n)

            setattr(mother, names[-1], scc_conv)
            #print('changed!')

    return net

