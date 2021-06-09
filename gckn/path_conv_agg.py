# -*- coding: utf-8 -*-
import torch
from gckn.gckn_fast.gckn_fast import path_conv_forward, path_conv_backward
from gckn.dynamic_pooling.pooling import dpooling_forward, dpooling_backward


MAXRAM = int(5e9)
# MAXRAM = int(5e7)
# MAXRAM = int(100000)

def get_batch_indices(array, batch_size):
    indices = [0]
    s = 0
    for i, v in enumerate(array):
        s += v.item()
        if s > batch_size:
            indices.append(i)
            s = v.item()
    indices.append(len(array))
    return indices

class PathConvAggregation(torch.autograd.Function):
    """Path extraction + convolution + aggregation
    features: n_nodes x path_size x hidden_size
    path_indices: n_paths x path_size
    kernel_size: n_nodes (sum = n_paths)
    pooling: {sum, mean, max}
    """
    @staticmethod
    def forward(ctx, features, path_indices, kernel_size, pooling='sum', kappa=torch.exp, d_kappa=torch.exp):
        batch_size = MAXRAM // (features.shape[-1] * features.element_size())
        indices = get_batch_indices(kernel_size, batch_size)
        batch_index = 0
        output = []
        active_indices = []
        n_paths_list = []
        # print(len(indices))
        for i in range(len(indices) - 1):
            batch_kernel_size = kernel_size[indices[i]:indices[i+1]]
            size = batch_kernel_size.sum().item()
            batch_path_indices = path_indices[batch_index: batch_index + size]
            embeded = path_conv_forward(batch_path_indices, features)
            embeded = kappa(embeded)
            embeded, active_index = dpooling_forward(embeded, batch_kernel_size, pooling=pooling)
            output.append(embeded)
            active_indices.append(active_index)
            n_paths_list.append(size)
            batch_index += size
        output = torch.cat(output)
        active_indices = torch.cat(active_indices)
        ctx.save_for_backward(path_indices, active_indices, features)
        ctx.indices = indices
        ctx.size = features.size()
        ctx.d_kappa = d_kappa
        ctx.pooling = pooling
        ctx.n_paths_list = n_paths_list
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        grad_input = grad_output.new_zeros(ctx.size)
        indices = ctx.indices
        path_indices, active_indices, features = ctx.saved_variables
        batch_index = 0
        grad_embed = grad_output.new_zeros(max(ctx.n_paths_list), ctx.size[-1])
        for i in range(len(indices) - 1):
            n_paths = ctx.n_paths_list[i]
            batch_path_indices = path_indices[batch_index: batch_index + n_paths]
            batch_index += n_paths
            # grad_embed = grad_output.new_zeros(n_paths, ctx.size[-1])
            grad_embed.zero_()
            dpooling_backward(grad_embed, grad_output[indices[i]:indices[i+1]], active_indices[indices[i]:indices[i+1]], ctx.pooling)
            embeded = path_conv_forward(batch_path_indices, features)
            embeded = ctx.d_kappa(embeded)
            grad_embed[:embeded.shape[0]].mul_(embeded)
            path_conv_backward(grad_input, grad_embed, batch_path_indices)
        return grad_input, None, None, None, None, None

from gckn.dynamic_pooling.pooling import dpooling_torch, dpooling
from gckn.gckn_fast.gckn_fast import path_conv, PathConv
def path_conv_agg_torch(features, path_indices, kernel_size, pooling='sum', kappa=torch.exp, d_kappa=None, mask=None):
    embeded = path_conv(path_indices, features)
    embeded = kappa(embeded)
    embeded = dpooling_torch(embeded, kernel_size, pooling)
    return embeded

def path_conv_agg(features, path_indices, kernel_size, pooling='sum', kappa=torch.exp, d_kappa=torch.exp, mask=None):
    ram_saving = MAXRAM <= (2 * path_indices.shape[0] * features.shape[-1] * features.element_size())
    if ram_saving and mask is None:
        return PathConvAggregation.apply(
            features, path_indices, kernel_size, pooling, kappa, d_kappa)
    embeded = PathConv.apply(path_indices, features)
    embeded = kappa(embeded)
    if mask is not None:
        embeded = embeded * mask.view(-1, 1)
    embeded = dpooling(embeded, kernel_size, pooling)
    return embeded

def test(cuda=False):
    torch.manual_seed(1234)

    pooling = 'sum'
    path_size = 5
    n_nodes = 100
    hidden_size = 32
    # n_paths = 10000
    max_length = 1000
    kappa = lambda x: torch.exp(2.*(x - 1.))
    d_kappa = lambda x: 2. * kappa(x)

    x = torch.randn(n_nodes, path_size+5, hidden_size)
    kernel_size = torch.randint(0, max_length, (n_nodes,))
    n_paths = kernel_size.sum().item()
    path_indices = torch.randint(0, n_nodes, (n_paths, path_size))
    print(path_indices.shape)
    if cuda:
        x = x.cuda()
        kernel_size = kernel_size.cuda()
        n_paths = n_paths.cuda()
        path_indices = path_indices.cuda()

    x.requires_grad_()

    print('start')
    out = PathConvAggregation.apply(x, path_indices, kernel_size, pooling, kappa, d_kappa)
    out1 = out.data
    out = out.mean()
    out.backward()
    grad1 = x.grad.data
    x.grad = None

    out = path_conv_agg(x, path_indices, kernel_size, pooling, kappa=kappa)
    out2 = out.data
    out = out.mean()
    out.backward()
    grad2 = x.grad.data
    # print(grad2)
    print(torch.max(torch.abs(out1 - out2)))
    print(torch.max(torch.abs(grad1 - grad2)))

    import time
    forward = 0
    backward = 0
    n_iter = 10
    for _ in range(n_iter):
        start = time.time()
        out = PathConvAggregation.apply(x, path_indices, kernel_size, pooling, kappa)
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
    n_iter = 10
    for _ in range(n_iter):
        start = time.time()
        out = path_conv_agg(x, path_indices, kernel_size, pooling, kappa)
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

if __name__ == '__main__':
    test()
