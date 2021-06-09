# -*- coding: utf-8 -*-
import math
import numpy as np
import os
import torch

from collections import defaultdict
from gckn.data import load_data, PathLoader
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

EPS = 1e-4


def normalize_(x, p=2, dim=-1, c2=1.):
    norm = x.norm(p=p, dim=dim, keepdim=True)
    x.div_(norm.clamp(min=EPS) / math.sqrt(c2))
    return x


EPS_ = 1e-6


def normalize(x, p=2, dim=-1, inplace=True):
    norm = x.norm(p=p, dim=dim, keepdim=True)
    if inplace:
        x.div_(norm.clamp(min=EPS_))
    else:
        x = x / norm.clamp(min=EPS_)
    return x


def init_kmeans(x, n_clusters, norm=1., n_local_trials=None, use_cuda=False):
    n_samples, n_features = x.size()
    clusters = torch.Tensor(n_clusters, n_features)
    if use_cuda:
        clusters = clusters.cuda()

    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(n_clusters))
    clusters[0] = x[np.random.randint(n_samples)]

    closest_dist_sq = 2 * (norm - clusters[[0]].mm(x.t()))
    closest_dist_sq = closest_dist_sq.view(-1)
    current_pot = closest_dist_sq.sum().item()

    for c in range(1, n_clusters):
        rand_vals = np.random.random_sample(n_local_trials).astype('float32') * current_pot
        rand_vals = np.minimum(rand_vals, current_pot * (1.0 - EPS))
        candidate_ids = np.searchsorted(closest_dist_sq.cumsum(-1).cpu(), rand_vals)
        distance_to_candidates = 2 * (norm - x[candidate_ids].mm(x.t()))

        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = torch.min(closest_dist_sq,
                                    distance_to_candidates[trial])
            new_pot = new_dist_sq.sum().item()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        clusters[c] = x[best_candidate]
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return clusters

def spherical_kmeans(x, n_clusters, max_iters=100, verbose=True,
                     init=None, eps=1e-4):
    """Spherical kmeans
    Args:
        x (Tensor n_samples x kmer_size x n_features): data points
        n_clusters (int): number of clusters
    """
    use_cuda = x.is_cuda
    if x.ndim == 3:
        n_samples, kmer_size, n_features = x.size()
    else:
        n_samples, n_features = x.size()
    if init == "kmeans++":
        print(init)
        if x.ndim == 3:
            clusters = init_kmeans(x.view(n_samples, -1), n_clusters,
                                   norm=kmer_size, use_cuda=use_cuda)
            clusters = clusters.view(n_clusters, kmer_size, n_features)
        else:
            clusters = init_kmeans(x, n_clusters, use_cuda=use_cuda)
    else:
        indices = torch.randperm(n_samples)[:n_clusters]
        if use_cuda:
            indices = indices.cuda()
        clusters = x[indices]

    prev_sim = np.inf

    for n_iter in range(max_iters):
        # assign data points to clusters
        cos_sim = x.view(n_samples, -1).mm(clusters.view(n_clusters, -1).t())
        tmp, assign = cos_sim.max(dim=-1)
        sim = tmp.mean()
        if (n_iter + 1) % 10 == 0 and verbose:
            print("Spherical kmeans iter {}, objective value {}".format(
                n_iter + 1, sim))

        # update clusters
        for j in range(n_clusters):
            index = assign == j
            if index.sum() == 0:
                # clusters[j] = x[random.randrange(n_samples)]
                idx = tmp.argmin()
                clusters[j] = x[idx]
                tmp[idx] = 1
            else:
                xj = x[index]
                c = xj.mean(0)
                clusters[j] = c / c.norm(dim=-1, keepdim=True).clamp(min=EPS)

        if torch.abs(prev_sim - sim)/(torch.abs(sim)+1e-20) < 1e-6:
            break
        prev_sim = sim
    return clusters

def block_diag(K, kernel_size):
    """
    input: a tensor: m x all_paths x out_size
    output: block diagonal tensor: m x all_paths x (len(kernel_size) * out_size)
    """
    idxs_stop = torch.cumsum(kernel_size, dim=0)
    idxs_start = idxs_stop - kernel_size
    arrs = [K[:, idx_start:idx_stop, :] for idx_start, idx_stop in zip(idxs_start, idxs_stop)]
    m, _, _ = K.shape
    shapes = torch.tensor([a.shape[1:] for a in arrs])
    out = torch.zeros([m] + torch.sum(shapes, dim=0).tolist(), dtype=arrs[0].dtype, device=arrs[0].device)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[:, r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc
    return out

def diag_to_compact(K, kernel_size, out_size):
    """
    input: a block diagonal tensor: m x all_paths x (len(kernel_size) * out_size)
    output: a compact tensor: m x all_paths x out_size
    """
    # import pdb; pdb.set_trace()
    arrs = []
    idxs_stop = torch.cumsum(kernel_size, dim=0)
    idxs_start = idxs_stop - kernel_size
    for i, (idx_start, idx_stop) in enumerate(zip(idxs_start, idxs_stop)):
        arrs.append(K[:, idx_start:idx_stop, out_size * i: out_size * (i + 1)])
    return torch.cat(arrs, dim=1)

def make_splits(dataset='MUTAG'):
    """
    This function makes stratified splits for nested cross-validation
    starting from cross-validation stratified splits.
    """
    outdir = '../experiments/dataset/{}/10fold_idx/inner_folds'.format(dataset)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if dataset in ['MUTAG', 'PROTEINS', 'PTC', 'NCI1']:
        graphs, _ = load_data(dataset, '../experiments/dataset', degree_as_tag=False)
    else:
        raise ValueError("make_splits not implemented for this dataset")
    train_fold_idxs = [np.loadtxt('../experiments/dataset/{}/10fold_idx/train_idx-{}.txt'.format(
            dataset, i)).astype(int) for i in range(1, 11)]
    kfold = StratifiedKFold(n_splits=10, shuffle=True)

    for i, train_fold_idx in enumerate(train_fold_idxs):
        data_loader = PathLoader([graphs[idx] for idx in train_fold_idx], 2, 32, True, dataset=dataset)
        data_loader.get_all_paths()
        train_fold_labels = data_loader.labels

        for j, split in enumerate(kfold.split(train_fold_idx.reshape(-1, 1), train_fold_labels)):
            inner_train_fold_idx, inner_val_fold_idx = split
            np.savetxt(os.path.join(outdir, 'train_idx-{}-{}.txt'.format(i + 1, j + 1)),
                    train_fold_idx[inner_train_fold_idx], fmt='%i')
            np.savetxt(os.path.join(outdir, 'val_idx-{}-{}.txt'.format(i + 1, j + 1)),
                    train_fold_idx[inner_val_fold_idx], fmt='%i')
    return


if __name__ == "__main__":
    # make_splits('NCI1')
    # graphs, _ = load_data('MUTAG', '../experiments/dataset', degree_as_tag=False)
    # g2l = Graph2Loader([3], 32, 'MUTAG', True)
    # output = g2l.transform(graphs)
    # print(output)
    pass
