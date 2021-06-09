# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
import torch.nn.functional as F
from . import ops
from .kernels import kernels, d_kernels
from .utils import EPS, normalize_, normalize, spherical_kmeans
from .dynamic_pooling.pooling import dpooling_torch as dpooling
from .path_conv_agg import path_conv_agg_torch as path_conv_agg

import numpy as np
from scipy import optimize
from sklearn.linear_model.base import LinearModel, LinearClassifierMixin


class PathLayer(nn.Module):
    def __init__(self, input_size, hidden_size, path_size=1,
                 kernel_func='exp', kernel_args=[0.5], pooling='mean',
                 aggregation=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.path_size = path_size

        self.pooling = pooling
        self.aggregation = aggregation and (path_size > 1)

        self.kernel_func = kernel_func
        if isinstance(kernel_args, (int, float)):
            kernel_args = [kernel_args]
        if kernel_func == 'exp':
            kernel_args = [1. / kernel_arg ** 2 for kernel_arg in kernel_args]
        self.kernel_args = kernel_args  # [kernel_arg / path_size for kernel_arg in kernel_args]
        self.kernel_func = kernels[kernel_func]
        self.kappa = lambda x: self.kernel_func(x, *self.kernel_args)
        d_kernel_func = d_kernels[kernel_func]
        self.d_kappa = lambda x: d_kernel_func(x, *self.kernel_args)

        self._need_lintrans_computed = True
        self.weight = nn.Parameter(
            torch.Tensor(path_size, hidden_size, input_size))

        if self.aggregation:
            self.register_buffer('lintrans',
                                 torch.Tensor(path_size, hidden_size,
                                              hidden_size)
                                 )
            self.register_buffer('divider',
                                 torch.arange(1., path_size + 1).view(-1, 1, 1)
                                 )
        else:
            self.register_buffer('lintrans',
                                 torch.Tensor(hidden_size, hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_size)
        for w in self.parameters():
            if w.dim() > 1:
                w.data.uniform_(-stdv, stdv)
        self.normalize_()

    def normalize_(self):
        normalize_(self.weight.data, dim=-1)

    def train(self, mode=True):
        super().train(mode)
        self._need_lintrans_computed = True

    def _compute_lintrans(self):
        if not self._need_lintrans_computed:
            return self.lintrans
        lintrans = torch.bmm(self.weight, self.weight.permute(0, 2, 1))
        if self.aggregation:
            lintrans = lintrans.cumsum(dim=0) / self.divider
        else:
            lintrans = lintrans.mean(dim=0)
        lintrans = self.kappa(lintrans)
        lintrans = ops.matrix_inverse_sqrt(lintrans)

        if not self.training:
            self._need_lintrans_computed = False
            self.lintrans.data.copy_(lintrans.data)
        return lintrans

    def forward(self, features, paths_indices, other_info):
        """
        features: n_nodes x (input_path_size) x input_size
        paths_indices: n_paths x path_size (values < n_nodes)
        output: n_nodes x ((input_path_size) x path_size) x input_size
        """
        # convolution
        self.normalize_()
        norms = features.norm(dim=-1, keepdim=True)
        # norms: n_nodes x (input_path_size) x 1
        # output = features / norms.clamp(min=EPS)
        output = torch.tensordot(features, self.weight, dims=[[-1], [-1]])
        output = output / norms.clamp(min=EPS).unsqueeze(2)
        n_nodes = output.shape[0]
        if output.ndim == 4:
            output = output.permute(0, 2, 1, 3).contiguous()
        # output: n_nodes x path_size x (input_path_size) x hidden_size

        # prepare masks
        mask = None
        if self.aggregation:
            mask = [None for _ in range(self.path_size)]
        if 'mask' in other_info and self.path_size > 1:
            mask = other_info['mask']

        output = output.view(n_nodes, self.path_size, -1)
        # output: n_nodes x path_size x (input_path_size x hidden_size)
        if self.aggregation:
            outputs = []
            for i in range(self.path_size):
                embeded = path_conv_agg(
                    output, paths_indices[i], other_info['n_paths'][i],
                    self.pooling, self.kappa, self.d_kappa, mask[i])
                outputs.append(embeded)
            output = torch.stack(outputs, dim=0)
            output = output.view(self.path_size, -1, self.hidden_size)
            # output: path_size x (n_nodes x (input_path_size)) x hidden_size
            output = norms.view(1, -1, 1) * output
        else:
            output = path_conv_agg(
                output, paths_indices[self.path_size - 1],
                other_info['n_paths'][self.path_size - 1],
                self.pooling, self.kappa, self.d_kappa, mask)
            # output: n_nodes x ((input_path_size) x hidden_size)
            output = output.view(n_nodes, -1, self.hidden_size)
            output = norms.view(n_nodes, -1, 1) * output
            # output: n_nodes x (input_path_size) x hidden_size

        lintrans = self._compute_lintrans()
        # linear transformation
        if self.aggregation:
            output = output.bmm(lintrans)
            # output = output.view(self.path_size, n_nodes, -1, self.hidden_size)
            output = output.permute(1, 0, 2)
            output = output.reshape(n_nodes, -1, self.hidden_size)
            output = output.contiguous()
        else:
            output = torch.tensordot(output, lintrans, dims=[[-1], [-1]])
        # output: n_nodes x ((input_path_size) x path_size) x hidden_size

        return output

    def sample_paths(self, features, paths_indices, n_sampling_paths=1000):
        """Sample paths for a given of features and paths
        features: n_nodes x (input_path_size) x input_size
        paths_indices: n_paths x path_size
        output: n_sampling_paths x path_size x input_size
        """
        paths_indices = paths_indices[self.path_size - 1]
        if self.path_size == 1:
            features = features.permute(1, 0, 2).reshape(-1, self.input_size)
            n_all_paths = features.shape[0]
            n_sampling_paths = min(n_all_paths, n_sampling_paths)
            indices = torch.randperm(n_all_paths)[:n_sampling_paths]
            paths = features[indices]
            return paths.view(n_sampling_paths, 1, self.input_size).contiguous()
        n_all_paths = paths_indices.shape[0]
        indices = torch.randperm(n_all_paths)[:min(n_all_paths, n_sampling_paths)]
        paths = F.embedding(paths_indices[indices], features)
        # paths: n_sampling_paths x path_size x (input_path_size) x input_size
        if paths.ndim == 4:
            paths = paths.permute(0, 2, 1, 3)
            paths = paths.reshape(-1, self.path_size, self.input_size)
            paths = paths[:min(paths.shape[0], n_sampling_paths)]
        return paths

    def unsup_train(self, paths, init=None):
        """Unsupervised training for path layer
        paths: n x path_size x input_size
        self.weight: path_size x hidden_size x input_size
        """
        # print(paths)
        print(paths.shape)
        normalize_(paths, dim=-1)
        weight = spherical_kmeans(paths, self.hidden_size, init='kmeans++')
        weight = weight.permute(1, 0, 2)
        self.weight.data.copy_(weight)

        self.normalize_()
        self._need_lintrans_computed = True


class NodePooling(nn.Module):
    def __init__(self, pooling='mean'):
        super().__init__()
        self.pooling = pooling

    def reset_parameters(self):
        pass

    def forward(self, features, other_info):
        """
        features: n_nodes x (input_path_size) x input_size
        output: n_graphs x input_size
        """
        # could normalizing the features before classification help?
        # why permuting?
        features = features.permute(0, 2, 1).contiguous()
        n_nodes = features.shape[0]
        output = dpooling(features.view(n_nodes, -1), other_info['n_nodes'],
                          self.pooling)

        return output


class Linear(nn.Linear, LinearModel, LinearClassifierMixin):
    def __init__(self, in_features, out_features, alpha=0.0, fit_bias=True):
        super(Linear, self).__init__(in_features, out_features, fit_bias)
        self.alpha = alpha
        self.fit_bias = fit_bias

    def forward(self, input, proba=False):
        out = super(Linear, self).forward(input)
        if proba:
            return out.sigmoid()
        return out

    def fit(self, x, y, criterion=None):
        use_cuda = self.weight.data.is_cuda
        if criterion is None:
            criterion = nn.BCEWithLogitsLoss()
        reduction = criterion.reduction
        criterion.reduction = 'sum'
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            x = torch.from_numpy(x)
            y = torch.from_numpy(y)
        if use_cuda:
            x = x.cuda()
            y = y.cuda()

        def eval_loss(w):
            w = w.reshape((self.out_features, -1))
            if self.weight.grad is not None:
                self.weight.grad = None
            if self.bias is None:
                self.weight.data.copy_(torch.from_numpy(w))
            else:
                if self.bias.grad is not None:
                    self.bias.grad = None
                self.weight.data.copy_(torch.from_numpy(w[:, :-1]))
                self.bias.data.copy_(torch.from_numpy(w[:, -1]))
            y_pred = self(x).squeeze_(-1)
            loss = criterion(y_pred, y)
            loss.backward()
            if self.alpha != 0.0:
                penalty = 0.5 * self.alpha * torch.norm(self.weight)**2
                loss = loss + penalty
            return loss.item()

        def eval_grad(w):
            dw = self.weight.grad.data
            if self.alpha != 0.0:
                dw.add_(self.alpha, self.weight.data)
            if self.bias is not None:
                db = self.bias.grad.data
                dw = torch.cat((dw, db.view(-1, 1)), dim=1)
            return dw.cpu().numpy().ravel().astype("float64")

        w_init = self.weight.data
        if self.bias is not None:
            w_init = torch.cat((w_init, self.bias.data.view(-1, 1)), dim=1)
        w_init = w_init.cpu().numpy().astype("float64")

        w = optimize.fmin_l_bfgs_b(
            eval_loss, w_init, fprime=eval_grad, maxiter=1000, disp=0)
        if isinstance(w, tuple):
            w = w[0]

        w = w.reshape((self.out_features, -1))
        self.weight.grad.data.zero_()
        if self.bias is None:
            self.weight.data.copy_(torch.from_numpy(w))
        else:
            self.bias.grad.data.zero_()
            self.weight.data.copy_(torch.from_numpy(w[:, :-1]))
            self.bias.data.copy_(torch.from_numpy(w[:, -1]))
        criterion.reduction = reduction

    def decision_function(self, x):
        x = torch.from_numpy(x)
        return self(x).data.numpy().ravel()

    def predict(self, x):
        return self.decision_function(x)

    def predict_proba(self, x):
        return self._predict_proba_lr(x)

    @property
    def coef_(self):
        return self.weight.data.numpy()

    @property
    def intercept_(self):
        return self.bias.data.numpy()
