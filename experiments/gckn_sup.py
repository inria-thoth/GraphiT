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
import torch.nn.functional as F

import pandas as pd
import argparse

from timeit import default_timer as timer


def load_args():
    parser = argparse.ArgumentParser(
        description='Supervised GCKN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--dataset', type=str, default="ZINC",
                        help='name of dataset')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--path-size', type=int, nargs='+', default=[4],
                        help='path sizes for layers')
    parser.add_argument('--hidden-size', type=int, nargs='+', default=[128],
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
    parser.add_argument('--batch-norm', action='store_true', help='use batch norm')
    args = parser.parse_args()
    if torch.cuda.is_available():
        args.use_cuda = True

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
        outdir = outdir + '/fold-{}'.format(args.seed)
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
    mse_loss = 0.0
    tic = timer()
    for data in data_loader.make_batch():
        features = data['features']
        paths_indices = data['paths']
        n_paths = data['n_paths']
        n_nodes = data['n_nodes']
        labels = data['labels']
        labels = labels.view(-1, 1)
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

        running_loss += loss.item() * size
        mse_loss += F.mse_loss(output, labels).item() * size

    toc = timer()
    n_sample = data_loader.n
    epoch_loss = running_loss / n_sample
    epoch_mse = mse_loss / n_sample

    print('Train mae loss: {:.4f} mse: {:.4f} time: {:.2f}s'.format(
          epoch_loss, epoch_mse, toc - tic))
    return epoch_loss, epoch_mse

def eval_epoch(model, data_loader, criterion, use_cuda=False):
    model.eval()

    running_loss = 0.0
    mae_loss = 0.0
    mse_loss = 0.0

    tic = timer()
    with torch.no_grad():
        for data in data_loader.make_batch(False):
            features = data['features']
            paths_indices = data['paths']
            n_paths = data['n_paths']
            n_nodes = data['n_nodes']
            labels = data['labels']
            labels = labels.view(-1, 1)
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

            running_loss += loss.item() * size
            mse_loss += F.mse_loss(output, labels).item() * size
            mae_loss += F.l1_loss(output, labels).item() * size
    toc = timer()

    n_sample = data_loader.n
    epoch_loss = running_loss / n_sample
    epoch_mae = mae_loss / n_sample
    epoch_mse = mse_loss / n_sample
    print('Val loss: {:.4f} mae: {:.4f} mse: {:.4f} time: {:.2f}s'.format(
          epoch_loss, epoch_mae, epoch_mse, toc - tic))
    return epoch_mae, epoch_mse


def main():
    global args
    args = load_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)

    data_path = '../dataset/ZINC'

    train_dset = datasets.ZINC(data_path, subset=True, split='train')
    val_dset = datasets.ZINC(data_path, subset=True, split='val')
    nclass = 1
    n_tags = 28

    graphloader = GraphLoader(args.path_size, args.batch_size, args.dataset,
                                args.walk)


    train_dset = convert_dataset(train_dset, n_tags)
    train_loader = graphloader.transform(train_dset)
    input_size = train_loader.input_size

    val_dset = convert_dataset(val_dset, n_tags)
    val_loader = graphloader.transform(val_dset)

    model = GCKNet(nclass, input_size, args.hidden_size, args.path_size,
       kernel_funcs=args.kernel_funcs,
       kernel_args_list=args.sigma,
       pooling=args.pooling,
       global_pooling=args.global_pooling,
       aggregation=args.aggregation,
       weight_decay=args.weight_decay,
       batch_norm=args.batch_norm)

    model.unsup_train(train_loader, n_sampling_paths=args.sampling_paths)

    if args.alternating:
        optimizer = optim.SGD(model.features.parameters(), lr=args.lr, momentum=0.9)
    else:
        optimizer = optim.AdamW([
                {'params': model.features.parameters()},
                {'params': model.classifier.parameters(), 'weight_decay': args.weight_decay}
                ], lr=args.lr)
    criterion = nn.L1Loss()
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    test_dset = datasets.ZINC(data_path, subset=True, split='test')
    test_dset = convert_dataset(test_dset, n_tags)
    test_loader = graphloader.transform(test_dset)

    if args.use_cuda:
        model.cuda()


    logs = defaultdict(list)
    best_val_loss = float('inf')
    best_model = None
    best_epoch = 0
    start_time = timer()
    print("Starting training....")
    for epoch in range(args.epochs):
        print('Epoch {}/{}, LR {:.6f}'.format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        train_loss, train_mse = train_epoch(epoch, model, train_loader, criterion, optimizer,
                lr_scheduler, args.alternating, use_cuda=args.use_cuda)
        val_loss, val_mse = eval_epoch(model, val_loader, criterion, args.use_cuda)
        test_loss, test_mse = eval_epoch(model, test_loader, criterion, args.use_cuda)
        lr_scheduler.step()

        logs['train_mae'].append(train_loss)
        logs['val_mae'].append(val_loss)
        logs['val_mse'].append(val_mse)
        logs['test_mae'].append(test_loss)
        logs['test_mse'].append(test_mse)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
    
    total_time = timer() - start_time
    print("best epoch: {} best val loss: {:.4f}".format(best_epoch, best_val_loss))
    model.load_state_dict(best_weights)

    print()
    print("Testing...")
    test_loss, test_mse_loss = eval_epoch(model, test_loader, criterion, args.use_cuda)

    print("test MAE loss {:.4f}".format(test_loss))

    if args.save_logs:
        logs = pd.DataFrame.from_dict(logs)
        logs.to_csv(args.outdir + '/logs.csv')
        results = {
            'test_mae': test_loss,
            'test_mse': test_mse_loss,
            'val_mae': best_val_loss,
            'best_epoch': best_epoch,
            'total_time': total_time,
        }
        results = pd.DataFrame.from_dict(results, orient='index')
        results.to_csv(args.outdir + '/results.csv',
                       header=['value'], index_label='name')
        torch.save(
            {'args': args,
            'state_dict': best_weights},
            args.outdir + '/model.pkl')


if __name__ == "__main__":
    main()
