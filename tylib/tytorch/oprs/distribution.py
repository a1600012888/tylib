import torch
from torch.autograd import Function

from scipy.stats import beta as scipy_beta


class BetaCDFFunction(Function):

    @staticmethod
    def forward(ctx, x:torch.Tensor, alpha:torch.Tensor, beta:torch.Tensor):
        ctx.alpha = alpha
        ctx.beta = beta

        ctx.save_for_backward(x, )

        #print(cpu_x.shape)
        cpu_x = x.cpu()
        cdfs = scipy_beta.cdf(cpu_x, alpha.cpu(), beta.cpu())

        cdfs = torch.tensor(cdfs).type_as(x)

        cdfs = cdfs.to(x.get_device())

        cdfs.requires_grad = x.requires_grad
        return cdfs



    @staticmethod
    def backward(ctx, grad_output):

        #print(ctx.needs_input_grad[0] == False, 'back?')
        #if ctx.needs_input_grad[0] == False:
        #    return None, None, None

        x, = ctx.saved_tensors
        avd_zero = 0 # maybe 1e-5
        #print('x: ', x)
        alpha, beta = ctx.alpha, ctx.beta
        pdf = torch.lgamma(alpha + beta)
        pdf = pdf - (torch.lgamma(alpha) + torch.lgamma(beta))

        #print('a, b:', alpha, beta)
        #print('pdf:', pdf)
        pdf = torch.exp(pdf)

        ones = torch.ones_like(x)
        pdf = pdf * torch.pow(x+avd_zero, alpha-1) * torch.pow(ones-x+avd_zero, beta-1)

        x_grad = pdf * grad_output
        #print(pdf)
        return x_grad, None, None



beta_cdf = BetaCDFFunction.apply



def test_beta_cdf():
    from torch.autograd import gradcheck

    #inp = ( torch.rand(1, 1, dtype=torch.double, requires_grad=True).cuda(),
    #        torch.tensor(0.5).cuda().double(), torch.tensor(0.5).cuda().double())
    inp = (torch.rand(1, 1, dtype=torch.float, requires_grad=True).cuda(),
           torch.tensor(0.5).cuda().float(), torch.tensor(0.5).cuda().float())

    print('x:', inp[0])
    test = gradcheck(beta_cdf, inp, eps=1e-6, atol=1e-4)

    print(test)

if __name__ == '__main__':
    test_beta_cdf()