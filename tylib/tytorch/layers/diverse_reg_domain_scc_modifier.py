import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import math

from ..oprs.distribution import beta_cdf
from ..oprs.index import unique_sort


class BetaCDFBatchShapingLoss(nn.Module):

    def __init__(self, alpha:torch.Tensor, beta:torch.Tensor):
        super(BetaCDFBatchShapingLoss, self).__init__()

        self.register_buffer('alpha', alpha)
        self.register_buffer('beta', beta)


    def __call__(self, vecs:torch.Tensor):


        #sorted_vecs = torch.unique(vecs, sorted=True, dim=0) # get ascending order
        #sorted_vecs, _ = torch.sort(vecs)
        sorted_vecs = unique_sort(vecs, 0)
        #sorted_vecs = vecs

        N = sorted_vecs.shape[0]

        idxs = torch.arange(1, N+1) / N

        idxs.unsqueeze_(dim=-1)
        idxs = idxs.type_as(sorted_vecs).to(sorted_vecs.get_device())


        cdfs = beta_cdf(sorted_vecs, self.alpha, self.beta)

        idxs = idxs.expand_as(cdfs)

        idxs = idxs - cdfs
        idxs = torch.flatten(idxs) #idxs.view(-1)

        loss = torch.dot(idxs, idxs) / N

        return loss


class DiverseRegDCConv2d(nn.Module):
    def __init__(self, embedding_in, num, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 alpha=0.5, beta=0.5, lamda=0.1):
        super(DiverseRegDCConv2d, self).__init__()
        self.num = num
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.register_buffer('alpha', torch.tensor(alpha))
        self.register_buffer('beta', torch.tensor(beta))
        self.register_buffer('lamda', torch.tensor(lamda))

        self.batch_shape_loss = BetaCDFBatchShapingLoss(self.alpha,
                                                        self.beta)

        self.routing_fc = nn.Linear(embedding_in, num)
        self.weight = Parameter(torch.Tensor(out_channels * in_channels * kernel_size * kernel_size // groups, num))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self._initialize_weights()
        self.register_buffer('inputs_se', None)
    def _initialize_weights(self):
        for name, m in self.named_modules():

            if isinstance(m, DiverseRegDCConv2d):
                c, h, w = m.in_channels, m.kernel_size, m.kernel_size
                nn.init.normal_(m.weight, 0, math.sqrt(1.0 / (c * h * w)))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.running_mean, 1)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.running_mean, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
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

        batchsize, channel, height, width = inputs.shape

        inputs_se = self.inputs_se # we need a pre_forward_hook to get this

        # add batch shaped loss
        if self.training:
            batch_shaped_loss = self.batch_shape_loss(inputs_se) * self.lamda
            batch_shaped_loss.backward(retain_graph=True)
            #rint(self.routing_fc.weight.grad.sum())

        weight = F.linear(inputs_se, self.weight)
        weight = weight.reshape(batchsize * self.out_channels, self.in_channels // self.groups, self.kernel_size,
                                self.kernel_size)

        inputs = inputs.reshape(1, batchsize * channel, height, width)
        outputs = F.conv2d(inputs, weight, None, self.stride, self.padding, self.dilation,
                           groups=self.groups * batchsize)
        height, width = outputs.shape[2:]
        outputs = outputs.reshape(batchsize, self.out_channels, height, width)
        if self.bias is not None:
            outputs = outputs + self.bias.reshape(1, -1, 1, 1)

        return outputs


def get_inputs_se(net:torch.nn.Module, inputs):
    #embedding_vec = inputs[-1]
    embedding_vec = net.embedding_vec
    #print(type(inputs), len(inputs), embedding_vec)
    for m in net.modules():
        #if m.__class_name__ == 'DCConv2d':
        if isinstance(m, DiverseRegDCConv2d):
            m.inputs_se = F.sigmoid(m.routing_fc(embedding_vec))

def clear_inputs_se(net:torch.nn.Module, inp, out):
    for m in net.modules():
        #if m.__class_name__ == 'DCConv2d':
        if isinstance(m, DiverseRegDCConv2d):
            m.inputs_se = None

def Conv2DScc(net:nn.Module, alpha=0.6, beta=0.4, lamda=0.2,
              num=4, in_num=2048):

    dic = dict(net.named_modules())
    for name, m in dic.items():
        if isinstance(m, nn.Conv2d):
            # print(num, m.in_channels, m.out_channels,
            #                    m.kernel_size, m.stride, m.padding,
            #                    m.dilation, m.groups, m.bias)
            scc_conv = DiverseRegDCConv2d(in_num, num, m.in_channels, m.out_channels,
                                          m.kernel_size[0], m.stride[0], m.padding[0],
                                          m.dilation[0], m.groups, m.bias, alpha, beta,
                                          lamda)
            #print('cc')

            names = name.split('.')
            mother = net
            for n in names[:-1]:
                mother = getattr(mother, n)

            setattr(mother, names[-1], scc_conv)
            #print('changed!')

    net.register_forward_pre_hook(get_inputs_se)
    net.register_forward_hook(clear_inputs_se)
    return net

