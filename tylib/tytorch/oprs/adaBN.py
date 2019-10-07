import torch
import torch.nn as nn

class AdaBN(nn.Module):
    '''
    2D domain aware Batch Normalization
    '''

    def __init__(self, num_domain, num_features, *args, **kwargs):
        super(AdaBN, self).__init__()
        self.num_domain = num_domain
        self.BN_buffer = nn.ModuleList()

        for i in range(self.num_domain):
            bn = nn.BatchNorm2d(num_features, *args, **kwargs)

            self.BN_buffer.append(bn)

        self.bn = self.BN_buffer[0]

    def forward(self, *input):
        out = self.bn(*input)
        return out

    def select_bn(self, idx:int):
        self.bn = self.BN_buffer[idx]



def BN2AdaBN(net:nn.Module, num_domain=4):
    import copy
    #dic = copy.deepcopy(dict(net.named_modules()))
    dic = dict(net.named_modules())
    for name, m in dic.items():
        if isinstance(m, nn.BatchNorm2d):
            # print(num, m.in_channels, m.out_channels,
            #                    m.kernel_size, m.stride, m.padding,
            #                    m.dilation, m.groups, m.bias)

            ada_bn = AdaBN(num_domain, m.num_features)

            names = name.split('.')
            mother = net
            for n in names[:-1]:
                mother = getattr(mother, n)

            setattr(mother, names[-1], ada_bn)
            #print('changed!')

    return net


def test_adabn():
    import torchvision
    net = torchvision.models.resnet50(pretrained=False)
    net = BN2AdaBN(net)
    print(net)

if __name__ == "__main__":
    test_adabn()