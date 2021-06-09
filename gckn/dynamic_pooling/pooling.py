# -*- coding: utf-8 -*-
import os
import torch
from gckn.dynamic_pooling import pooling_cpu
if torch.cuda.is_available():
    try:
        from gckn.dynamic_pooling import pooling_cuda
    except:
        pass


def dpooling_forward(input, kernel_size, pooling='sum'):
    kernel_size = kernel_size.cumsum(0)
    active_indices = kernel_size
    if pooling == 'max':
        if input.is_cuda:
            output, active_indices = pooling_cuda.max_forward(input, kernel_size)
        else:
            output, active_indices = pooling_cpu.max_forward(input, kernel_size)
    elif pooling == 'sum':
        if input.is_cuda:
            output = pooling_cuda.sum_forward(input, kernel_size, False)
        else:
            output = pooling_cpu.sum_forward(input, kernel_size, False)
    elif pooling == 'mean':
        if input.is_cuda:
            output = pooling_cuda.sum_forward(input, kernel_size, True)
        else:
            output = pooling_cpu.sum_forward(input, kernel_size, True)
    return output, active_indices

def dpooling_backward(grad_input, grad_output, indices, pooling='sum'):
    if pooling == 'max':
        if grad_output.is_cuda:
            pooling_cuda.max_backward(grad_input, grad_output, indices)
        else:
            pooling_cpu.max_backward(grad_input, grad_output, indices)
    elif pooling == 'sum':
        if grad_output.is_cuda:
            pooling_cuda.sum_backward(grad_input, grad_output, indices, False)
        else:
            pooling_cpu.sum_backward(grad_input, grad_output, indices, False)
    elif pooling == 'mean':
        if grad_output.is_cuda:
            pooling_cuda.sum_backward(grad_input, grad_output, indices, True)
        else:
            pooling_cpu.sum_backward(grad_input, grad_output, indices, True)

class DPoolingMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, kernel_size):
        kernel_size = kernel_size.cumsum(0)
        if input.is_cuda:
            output, active_indices = pooling_cuda.max_forward(input, kernel_size)
        else:
            output, active_indices = pooling_cpu.max_forward(input, kernel_size)
        ctx.save_for_backward(active_indices)
        ctx.size = input.shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.new_zeros(ctx.size)
        if grad_output.is_cuda:
            pooling_cuda.max_backward(grad_input, grad_output.contiguous(), *ctx.saved_variables)
        else:
            pooling_cpu.max_backward(grad_input, grad_output.contiguous(), *ctx.saved_variables)
        return grad_input, None

class DPoolingSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, kernel_size, mean=False):
        kernel_size = kernel_size.cumsum(0)
        if input.is_cuda:
            output = pooling_cuda.sum_forward(input, kernel_size, mean)
        else:
            output = pooling_cpu.sum_forward(input, kernel_size, mean)
        ctx.save_for_backward(kernel_size)
        ctx.mean = mean
        ctx.size = input.shape
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.new_zeros(ctx.size)
        if grad_output.is_cuda:
            pooling_cuda.sum_backward(grad_input, grad_output.contiguous(), *ctx.saved_variables, ctx.mean)
        else:
            pooling_cpu.sum_backward(grad_input, grad_output.contiguous(), *ctx.saved_variables, ctx.mean)
        return grad_input, None, None

def dpooling(input, kernel_size, pooling='sum'):
    if pooling == 'sum':
        return DPoolingSum.apply(input, kernel_size, False)
    elif pooling == 'mean':
        return DPoolingSum.apply(input, kernel_size, True)
    elif pooling == 'max':
        return DPoolingMax.apply(input, kernel_size)
    else:
        raise ValueError('Not implemented!')

def dpooling_max_pad(input, kernel_size):
    output = torch.split(input, list(kernel_size))
    output = torch.nn.utils.rnn.pad_sequence(output)
    output,_ = output.max(dim=0)
    return output

def dpooling_torch(input, kernel_size, pooling='sum'):
    if pooling == 'max':
        return dpooling_max_pad(input, kernel_size)
    all_paths = input.shape[0]
    row = torch.arange(len(kernel_size), device=kernel_size.device).repeat_interleave(kernel_size, dim=0)
    col = torch.arange(all_paths, device=kernel_size.device)
    indices = torch.stack([row, col], dim=0)
    if pooling == 'sum':
        data = torch.ones(all_paths, device=kernel_size.device)
    elif pooling == 'mean':
        data = 1. / kernel_size.float()
        data = data.repeat_interleave(kernel_size, dim=0)
    else:
        raise ValueError('Not implemented!')
    pool_matrix = torch.sparse.FloatTensor(
        indices, data, torch.Size([len(kernel_size), all_paths]))

    return pool_matrix.mm(input)

def test(cuda=False):
    torch.manual_seed(1234)
    pooling = 'sum'
    size = 500
    max_length = 100
    hidden_size = 128
    kernel_size = torch.randint(0, max_length, (size,))
    x = torch.rand(kernel_size.sum(), hidden_size)
    # kernel_size = torch.LongTensor([4, 5, 1])
    if cuda:
        x = x.cuda()
        kernel_size = kernel_size.cuda()

    x.requires_grad_()
    print('start')
    out = dpooling(x, kernel_size, pooling=pooling)
    out1 = out.data
    out = out.mean()
    out.backward()
    grad1 = x.grad.data

    x.grad = None
    out = dpooling_torch(x, kernel_size, pooling=pooling)
    out2 = out.data
    out = out.mean()
    out.backward()
    grad2 = x.grad.data
    # print(x)
    # print(kernel_size)
    # print(out1)
    # print(out2)
    # print(grad1)
    # print(grad2)

    print(torch.max(torch.abs(out1 - out2)))
    print(torch.max(torch.abs(grad1 - grad2)))

    import time
    forward = 0
    backward = 0
    n_iter = 100
    for _ in range(n_iter):
        start = time.time()
        out = dpooling(x, kernel_size, pooling=pooling)
        if cuda:
            torch.cuda.synchronize()
        forward += time.time() - start

        out = out.mean()
        start = time.time()
        out.backward()
        if cuda:
            torch.cuda.synchronize()
        backward += time.time() - start

    print('Mine Forward: {:.3f} ms | Backward {:.3f} ms'.format(forward * 1e3/n_iter, backward * 1e3/n_iter))

    import time
    forward = 0
    backward = 0
    n_iter = 100
    for _ in range(n_iter):
        start = time.time()
        out = dpooling_torch(x, kernel_size, pooling=pooling)
        if cuda:
            torch.cuda.synchronize()
        forward += time.time() - start

        out = out.mean()
        start = time.time()
        out.backward()
        if cuda:
            torch.cuda.synchronize()
        backward += time.time() - start

    print('Pytorch Forward: {:.3f} ms | Backward {:.3f} ms'.format(forward * 1e3/n_iter, backward * 1e3/n_iter))


if __name__ == "__main__":
    # test(cuda=False)
    # import pdb; pdb.set_trace()
    input = torch.rand(10, 5)
    attn_weight = torch.rand(10, 3, 2)
    kernel_size = torch.tensor([4, 3, 3])
    print(dpooling_ot(input, attn_weight, kernel_size))
