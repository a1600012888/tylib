import torch
from torch.autograd import Function


class UniqueSort(Function):

    @staticmethod
    def forward(ctx, x, dim=0):
        # the backward only implements the "dim=0" case!!(because we need reshape)

        sorted_x, idx, couts = torch.unique(x, sorted=True, return_inverse=True, return_counts=True, dim=dim)

        #print(sorted_x, idx, couts)
        #print(idx.shape, couts.shape)

        ctx.save_for_backward(x, idx, couts)

        return sorted_x

    @staticmethod
    def backward(ctx, grad_outputs):
        # note all unsequeeze and expand is for "dim=0" case

        x, idx, couts = ctx.saved_tensors

        inv_c = 1.0 / couts.type_as(grad_outputs)
        inv_c = inv_c.unsqueeze(dim=-1)

        #print(grad_outputs.shape, 'grad out shape')
        grad_source = grad_outputs * inv_c


        # following unsquezze and expand only for "dim=0"

        idx = idx.unsqueeze(dim=-1)
        idx = idx.expand_as(x)
        #print(idx, 'idx')
        #print(grad_source, 'grad source')

        grad = torch.gather(grad_source, dim=0, index=idx)

        #print(grad, 'grad')
        return grad, None

unique_sort = UniqueSort.apply

def test_unique_sort():
    from torch.autograd import gradcheck

    #inp = ( torch.rand(1, 1, dtype=torch.double, requires_grad=True).cuda(),
    #        torch.tensor(0.5).cuda().double(), torch.tensor(0.5).cuda().double())

    x = torch.tensor(
        [[1.0,2.0,3.0],
         [1.0,2.0,3.0],
         [2.0,3.0,4.0],
         [3.0,4.0,5.0],
         [2.0,3.0,4.0],
         ],
    dtype=torch.float, requires_grad=True).cuda()

    inp = (x, )

    #print('x:', inp[0])
    test = gradcheck(unique_sort, inp, eps=1e-6, atol=1e-4)

    print(test)


if __name__ == '__main__':
    test_unique_sort()