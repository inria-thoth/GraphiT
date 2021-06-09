# -*- coding: utf-8 -*-
import torch 


class MatrixInverseSqrt(torch.autograd.Function):
    """Matrix inverse square root for a symmetric definite positive matrix
    """
    @staticmethod
    def forward(ctx, input, eps=1e-2):
        dim = input.dim()
        ctx.dim = dim
        use_cuda = input.is_cuda
        if input.size(0) < 300:
            input = input.cpu()
        e, v = torch.symeig(input, eigenvectors=True)
        if use_cuda and input.size(0) < 300:
            e = e.cuda()
            v = v.cuda()
        e = e.clamp(min=0)
        e_sqrt = e.sqrt_().add_(eps)
        ctx.save_for_backward(e_sqrt, v)
        e_rsqrt = e_sqrt.reciprocal()

        if dim > 2:
            output = v.bmm(v.permute(0, 2, 1) * e_rsqrt.unsqueeze(-1))
        else:
            output = v.mm(v.t() * e_rsqrt.view(-1, 1))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        e_sqrt, v = ctx.saved_variables
        if ctx.dim > 2:
            ei = e_sqrt.unsqueeze(1).expand_as(v)
            ej = e_sqrt.unsqueeze(-1).expand_as(v)
        else:
            ei = e_sqrt.expand_as(v)
            ej = e_sqrt.view(-1, 1).expand_as(v)
        f = torch.reciprocal((ei + ej) * ei * ej)
        if ctx.dim > 2:
            vt = v.permute(0, 2, 1)
            grad_input = -v.bmm((f*(vt.bmm(grad_output.bmm(v)))).bmm(vt))
        else:
            grad_input = -v.mm((f*(v.t().mm(grad_output.mm(v)))).mm(v.t()))
        return grad_input, None


def matrix_inverse_sqrt(input, eps=1e-2):
    """Wrapper for MatrixInverseSqrt"""
    return MatrixInverseSqrt.apply(input, eps)


if __name__ == "__main__":
    torch.manual_seed(0)
    # x = torch.rand(1, 3, 3)
    # x = torch.bmm(x, x.permute(0, 2, 1))
    x = torch.rand(3, 3)
    x = torch.mm(x, x.t())
    x.unsqueeze_(0)
    x = x.repeat(5, 1, 1)
    x.requires_grad_()
    out = matrix_inverse_sqrt(x)
    print(out)
    out = out.sum()
    out.backward()
    print(x.grad)
