# -*- coding: utf-8 -*-
import numpy as np
import os
import copy
import torch
from collections import defaultdict

from gckn.data import load_data, GraphLoader
from gckn.models import GCKNet
from gckn.loss import LOSS
from torch import nn, optim
from data_utils import convert_dataset
from torch_geometric import datasets

import pandas as pd
import argparse

from timeit import default_timer as timer


def load_args():
    parser = argparse.ArgumentParser(
        description='Supervised GCKN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--dataset', type=str, default="MUTAG",
                        help='name of dataset')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--path-size', type=int, nargs='+', default=[3],
                        help='path sizes for layers')
    parser.add_argument('--hidden-size', type=int, nargs='+', default=[32],
                        help='number of filters for layers')
    parser.add_argument('--pooling', type=str, default='sum',
                        help='local path pooling for each node')
    parser.add_argument('--global-pooling', type=str, default='sum',
                        help='global node pooling for each graph')
    parser.add_argument('--aggregation', action='store_true',
                        help='aggregate all path features until path size')
    parser.add_argument('--kernel-funcs', type=str, nargs='+', default=None,
                        help='kernel functions')
    parser.add_argument('--sigma', type=float, nargs='+', default=[0.5],
                        help='sigma of expo (Gaussian) kernels for layers')
    parser.add_argument('--sampling-paths', type=int, default=300000,
                        help='number of paths to sample for unsup training')
    parser.add_argument('--weight-decay', type=float, default=1e-04,
                        help='weight decay for classifier')
    parser.add_argument('--alternating', action='store_true',
                        help='use alternating training')
    parser.add_argument('--walk', action='store_true',
                        help='use walk instead of path')
    parser.add_argument('--use-cuda', action='store_true',
                        help='use cuda or not')
    parser.add_argument('--outdir', type=str, default='',
                        help='output path')
    parser.add_argument('--fold-idx', type=int, default=1,
                        help='indices for the train/test datasets')
    parser.add_argument('--test', action='store_true', help='train on full train+val dataset')
    args = parser.parse_args()
    if torch.cuda.is_available():
        args.use_cuda = True

    args.continuous = False
    args.degree_as_tag = False

    args.save_logs = False
    if args.outdir != '':
        args.save_logs = True
        outdir = args.outdir
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        outdir = outdir + '/gckn_sup'
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        outdir = outdir + '/{}'.format(args.dataset)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        if args.aggregation:
            outdir = outdir + '/aggregation'
            if not os.path.exists(outdir):
                try:
                    os.makedirs(outdir)
                except Exception:
                    pass
        outdir = outdir + '/{}_{}_{}_{}_{}_{}_{}'.format(
            args.path_size, args.hidden_size, args.pooling,
            args.global_pooling, args.sigma, args.weight_decay,
            args.lr)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        outdir = outdir + '/fold-{}'.format(args.fold_idx)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        args.outdir = outdir
    return args


def train_epoch(epoch, model, data_loader, criterion, optimizer,
        lr_scheduler, alternating, use_cuda=False):

    if alternating or epoch == 0:
        model.eval()
        model.unsup_train_classifier(data_loader, criterion,
                                     use_cuda=use_cuda)
    model.train()

    running_loss = 0.0
    running_acc = 0.0
    tic = timer()
    for data in data_loader.make_batch():
        features = data['features']
        paths_indices = data['paths']
        n_paths = data['n_paths']
        n_nodes = data['n_nodes']
        labels = data['labels']
        size = len(n_nodes)
        if use_cuda:
            features = features.cuda()
            if isinstance(n_paths, list):
                paths_indices = [p.cuda() for p in paths_indices]
                n_paths = [p.cuda() for p in n_paths]
            else:
                paths_indices = paths_indices.cuda()
                n_paths = n_paths.cuda()
            n_nodes = n_nodes.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()
        output = model(
            features, paths_indices, {'n_paths': n_paths, 'n_nodes': n_nodes})
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        pred = output.data.argmax(dim=1)
        running_loss += loss.item() * size
        running_acc += torch.sum(pred == labels).item()

    toc = timer()
    n_sample = data_loader.n
    epoch_loss = running_loss / n_sample
    epoch_acc = running_acc / n_sample

    print('Train loss: {:.4f} Acc: {:.4f} time: {:.2f}s'.format(
          epoch_loss, epoch_acc, toc - tic))
    return epoch_acc, epoch_loss

def eval_epoch(model, data_loader, criterion, use_cuda=False):
    model.eval()

    running_loss = 0.0
    running_acc = 0.0

    tic = timer()
    with torch.no_grad():
        for data in data_loader.make_batch(False):
            features = data['features']
            paths_indices = data['paths']
            n_paths = data['n_paths']
            n_nodes = data['n_nodes']
            labels = data['labels']
            size = len(n_nodes)
            if use_cuda:
                features = features.cuda()
                if isinstance(n_paths, list):
                    paths_indices = [p.cuda() for p in paths_indices]
                    n_paths = [p.cuda() for p in n_paths]
                else:
                    paths_indices = paths_indices.cuda()
                    n_paths = n_paths.cuda()
                n_nodes = n_nodes.cuda()
                labels = labels.cuda()

            output = model(
                features, paths_indices, {'n_paths': n_paths, 'n_nodes': n_nodes})
            loss = criterion(output, labels)

            pred = output.data.argmax(dim=-1)
            running_acc += torch.sum(pred == labels).item()

            running_loss += loss.item() * size
    toc = timer()

    n_sample = data_loader.n
    epoch_loss = running_loss / n_sample
    epoch_acc = running_acc / n_sample
    print('Val loss: {:.4f} Acc: {:.4f} time: {:.2f}s'.format(
          epoch_loss, epoch_acc, toc - tic))
    return epoch_acc, epoch_loss


def main():
    global args
    args = load_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)

    data_path = '../dataset/TUDataset'

    dset_name = args.dataset
    if args.dataset == 'PTC':
        dset_name = 'PTC_MR'

    dset = datasets.TUDataset(data_path, dset_name)
    nclass = dset.num_classes
    n_tags = None

    graphloader = GraphLoader(args.path_size, args.batch_size, args.dataset,
                                args.walk)

    inner_idx = 1
    idx_path = '../dataset/fold-idx/{}/inner_folds/{}-{}-{}.txt'
    test_idx_path = '../dataset/fold-idx/{}/test_idx-{}.txt'
    train_fold_idx = torch.from_numpy(np.loadtxt(
        idx_path.format(args.dataset, 'train_idx', args.fold_idx, inner_idx)).astype(int))
    val_fold_idx = torch.from_numpy(np.loadtxt(
        idx_path.format(args.dataset, 'val_idx', args.fold_idx, inner_idx)).astype(int))
    test_fold_idx = torch.from_numpy(np.loadtxt(
        test_idx_path.format(args.dataset, args.fold_idx)).astype(int))

    if args.test:
        train_fold_idx = torch.cat((train_fold_idx, val_fold_idx), 0)

    train_dset = convert_dataset(dset[train_fold_idx], n_tags)
    train_loader = graphloader.transform(train_dset)
    input_size = train_loader.input_size

    val_dset = convert_dataset(dset[val_fold_idx], n_tags)
    val_loader = graphloader.transform(val_dset)

    model = GCKNet(nclass, input_size, args.hidden_size, args.path_size,
       kernel_funcs=args.kernel_funcs,
       kernel_args_list=args.sigma,
       pooling=args.pooling,
       global_pooling=args.global_pooling,
       aggregation=args.aggregation,
       weight_decay=args.weight_decay)

    model.unsup_train(train_loader, n_sampling_paths=args.sampling_paths)

    if args.alternating:
        optimizer = optim.SGD(model.features.parameters(), lr=args.lr, momentum=0.9)
    else:
        optimizer = optim.Adam([
                {'params': model.features.parameters()},
                {'params': model.classifier.parameters(), 'weight_decay': args.weight_decay}
                ], lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    test_dset = convert_dataset(dset[test_fold_idx], n_tags)
    test_loader = graphloader.transform(test_dset)

    if args.use_cuda:
        model.cuda()


    logs = defaultdict(list)
    best_val_acc = 0
    best_model = None
    best_epoch = 0
    start_time = timer()
    print("Starting training....")
    for epoch in range(args.epochs):
        print('Epoch {}/{}, LR {:.6f}'.format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        train_acc, train_loss = train_epoch(epoch, model, train_loader, criterion, optimizer,
                lr_scheduler, args.alternating, use_cuda=args.use_cuda)
        val_acc, val_loss = eval_epoch(model, val_loader, criterion, args.use_cuda)
        test_acc, test_loss = eval_epoch(model, test_loader, criterion, args.use_cuda)
        lr_scheduler.step()

        logs['train_loss'].append(train_loss)
        logs['val_acc'].append(val_acc)
        logs['val_loss'].append(val_loss)
        logs['test_acc'].append(test_acc)
        logs['test_loss'].append(test_loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
    
    total_time = timer() - start_time
    print("best epoch: {} best val acc: {:.4f}".format(best_epoch, best_val_acc))
    model.load_state_dict(best_weights)

    print()
    print("Testing...")
    test_acc, test_loss = eval_epoch(model, test_loader, criterion, args.use_cuda)

    print("test Acc {:.4f}".format(test_acc))

    if args.save_logs:
        logs = pd.DataFrame.from_dict(logs)
        logs_suffix = '_test' if args.test else ''
        logs.to_csv(args.outdir + '/logs{}.csv'.format(logs_suffix))
        results = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'total_time': total_time,
        }
        results = pd.DataFrame.from_dict(results, orient='index')
        results.to_csv(args.outdir + '/results{}.csv'.format(logs_suffix),
                       header=['value'], index_label='name')
        torch.save(
            {'args': args,
            'state_dict': best_weights},
            args.outdir + '/model{}.pkl'.format(logs_suffix))


if __name__ == "__main__":
    main()
