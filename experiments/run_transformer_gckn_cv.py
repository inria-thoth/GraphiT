# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
import copy
import pandas as pd
from collections import defaultdict
import torch


import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric import datasets
from transformer.models import DiffGraphTransformer, GraphTransformer
from transformer.data import GraphDataset
from transformer.position_encoding import LapEncoding, POSENCODINGS
from transformer.gckn_pe import GCKNEncoding
from transformer.utils import count_parameters
from timeit import default_timer as timer
from torch import nn, optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_args():
    parser = argparse.ArgumentParser(
        description='Transformer baseline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--dataset', type=str, default="NCI1",
                        help='name of dataset')
    parser.add_argument('--nb-heads', type=int, default=4)
    parser.add_argument('--nb-layers', type=int, default=3)
    parser.add_argument('--dim-hidden', type=int, default=64)
    parser.add_argument('--pos-enc', choices=[None,
                        'diffusion', 'pstep', 'adj'], default=None)
    parser.add_argument('--gckn-dim', type=int, default=32, help='dimension for laplacian PE')
    parser.add_argument('--gckn-path', type=int, default=5, help='path size for gckn')
    parser.add_argument('--gckn-sigma', type=float, default=0.6)
    parser.add_argument('--gckn-pooling', default='sum', choices=['mean', 'sum'])
    parser.add_argument('--gckn-agg', action='store_false', help='do not use aggregated GCKN features')
    parser.add_argument('--gckn-normalize', action='store_false', help='do not normalize gckn features')
    parser.add_argument('--p', type=int, default=1, help='p step random walk kernel')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='bandwidth for the diffusion kernel')
    parser.add_argument('--normalization', choices=[None, 'sym', 'rw'], default='sym',
                        help='normalization for Laplacian')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--outdir', type=str, default='',
                        help='output path')
    parser.add_argument('--warmup', type=int, default=2000)
    parser.add_argument('--batch-norm', action='store_true', help='use batch norm instead of layer norm')
    parser.add_argument('--zero-diag', action='store_true', help='zero diagonal for PE matrix')
    parser.add_argument('--fold-idx', type=int, default=1,
                        help='indices for the train/test datasets')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--test', action='store_true', help='train on full train+val dataset')
    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()

    args.save_logs = False
    if args.outdir != '':
        args.save_logs = True
        outdir = args.outdir
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        outdir = outdir + '/transformer'
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
        if args.zero_diag:
            outdir = outdir + '/zero_diag'
            if not os.path.exists(outdir):
                try:
                    os.makedirs(outdir)
                except Exception:
                    pass
        lapdir = 'gckn_{}_{}_{}_{}_{}_{}'.format(args.gckn_path, args.gckn_dim, args.gckn_sigma, args.gckn_pooling,
            args.gckn_agg, args.gckn_normalize) 
        outdir = outdir + '/{}'.format(lapdir)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        bn = 'BN' if args.batch_norm else 'LN'
        outdir = outdir + '/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
            args.weight_decay, args.dropout, args.lr, args.nb_layers, args.nb_heads, args.dim_hidden, bn,
            args.pos_enc, args.normalization, args.p, args.beta
        )
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


def train_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, use_cuda=False):
    model.train()

    running_loss = 0.0
    running_acc = 0.0

    tic = timer()
    for i, (data, mask, pe, lap_pe, degree, labels) in enumerate(loader):
        labels = labels.view(-1)
        if args.warmup is not None:
            iteration = epoch * len(loader) + i
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_scheduler(iteration)

        if use_cuda:
            data = data.cuda()
            mask = mask.cuda()
            if pe is not None:
                pe = pe.cuda()
            if lap_pe is not None:
                lap_pe = lap_pe.cuda()
            if degree is not None:
                degree = degree.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        output = model(data, mask, pe, lap_pe, degree)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        pred = output.data.argmax(dim=1)
        running_loss += loss.item() * len(data)
        running_acc += torch.sum(pred == labels).item()

    toc = timer()
    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    epoch_acc = running_acc / n_sample
    print('Train loss: {:.4f} Acc: {:.4f} time: {:.2f}s'.format(
          epoch_loss, epoch_acc, toc - tic))
    return epoch_loss


def eval_epoch(model, loader, criterion, use_cuda=False):
    model.eval()

    running_loss = 0.0
    running_acc = 0.0

    tic = timer()
    with torch.no_grad():
        for data, mask, pe, lap_pe, degree, labels in loader:
            labels = labels.view(-1)

            if use_cuda:
                data = data.cuda()
                mask = mask.cuda()
                if pe is not None:
                    pe = pe.cuda()
                if lap_pe is not None:
                    lap_pe = lap_pe.cuda()
                if degree is not None:
                    degree = degree.cuda()
                labels = labels.cuda()

            output = model(data, mask, pe, lap_pe, degree)
            loss = criterion(output, labels)

            pred = output.data.argmax(dim=-1)
            running_acc += torch.sum(pred == labels).item()

            running_loss += loss.item() * len(data)
    toc = timer()

    n_sample = len(loader.dataset)
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
    if args.dataset == "PTC":
        dset_name = 'PTC_MR'

    dset = datasets.TUDataset(data_path, dset_name)
    n_tags = None#dset[0].num_node_features
    nb_class = dset.num_classes
    idx_path = '../dataset/fold-idx/{}/inner_folds/{}-{}-{}.txt'
    test_idx_path = '../dataset/fold-idx/{}/test_idx-{}.txt'

    inner_idx = 1
    train_fold_idx = torch.from_numpy(np.loadtxt(
        idx_path.format(args.dataset, 'train_idx', args.fold_idx, inner_idx)).astype(int))
    val_fold_idx = torch.from_numpy(np.loadtxt(
        idx_path.format(args.dataset, 'val_idx', args.fold_idx, inner_idx)).astype(int))
    test_fold_idx = torch.from_numpy(np.loadtxt(
        test_idx_path.format(args.dataset, args.fold_idx)).astype(int))

    if not os.path.exists("../cache/pe/{}".format(args.dataset)):
        try:
            os.makedirs("../cache/pe/{}".format(args.dataset))
        except Exception:
            pass
    gckn_pos_enc_path = '../cache/pe/{}/gckn_fold{}_{}_{}_{}_{}_{}_{}.pkl'.format(
        args.dataset, args.fold_idx, args.gckn_path, args.gckn_dim, args.gckn_sigma, args.gckn_pooling,
        args.gckn_agg, args.gckn_normalize)
    gckn_pos_encoder = GCKNEncoding(
        gckn_pos_enc_path, args.gckn_dim, args.gckn_path, args.gckn_sigma, args.gckn_pooling,
        args.gckn_agg, args.gckn_normalize)
    print('GCKN Position encoding')
    gckn_pos_enc_values = gckn_pos_encoder.apply_to(
        dset[train_fold_idx], dset[torch.cat((val_fold_idx, test_fold_idx), 0)], batch_size=64, n_tags=n_tags)
    gckn_dim = gckn_pos_encoder.pos_enc_dim
    del gckn_pos_encoder

    print(len(gckn_pos_enc_values))

    if args.test:
        train_fold_idx = torch.cat((train_fold_idx, val_fold_idx), 0)

    train_dset = GraphDataset(dset[train_fold_idx], n_tags, degree=True)
    input_size = train_dset.input_size()
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=False, collate_fn=train_dset.collate_fn())
    print(len(train_dset))
    print(train_dset[0])

    val_dset = GraphDataset(dset[val_fold_idx], n_tags, degree=True)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dset.collate_fn())

    pos_encoder = None
    if args.pos_enc is not None:
        pos_encoding_method = POSENCODINGS.get(args.pos_enc, None)
        pos_encoding_params_str = ""
        if args.pos_enc == 'diffusion':
            pos_encoding_params = {
                'beta': args.beta
            }
            pos_encoding_params_str = args.beta
        elif args.pos_enc == 'pstep':
            pos_encoding_params = {
                'beta': args.beta,
                'p': args.p
            }
            pos_encoding_params_str = "{}_{}".format(args.p, args.beta)
        else:
            pos_encoding_params = {}

        if pos_encoding_method is not None:
            pos_cache_path = '../cache/pe/{}/{}_{}_{}.pkl'.format(
                args.dataset, args.pos_enc, args.normalization, pos_encoding_params_str)
            pos_encoder = pos_encoding_method(
                pos_cache_path, normalization=args.normalization, zero_diag=args.zero_diag, **pos_encoding_params)

        print("Position encoding...")
        pos_encoder.apply_to(dset, split='all')
        train_dset.pe_list = [dset.pe_list[i] for i in train_fold_idx]
        val_dset.pe_list = [dset.pe_list[i] for i in val_fold_idx]
    
    train_dset.lap_pe_list = gckn_pos_enc_values[:len(train_dset)]
    if args.test:
        val_dset.lap_pe_list = gckn_pos_enc_values[len(train_dset) - len(val_dset):len(train_dset)]
    else:
        val_dset.lap_pe_list = gckn_pos_enc_values[len(train_dset):len(train_dset) + len(val_dset)]

    if args.pos_enc is not None:
        model = DiffGraphTransformer(in_size=input_size,
                                     nb_class=nb_class,
                                     d_model=args.dim_hidden,
                                     dim_feedforward=2*args.dim_hidden,
                                     dropout=args.dropout,
                                     nb_heads=args.nb_heads,
                                     nb_layers=args.nb_layers,
                                     batch_norm=args.batch_norm,
                                     lap_pos_enc=True,
                                     lap_pos_enc_dim=gckn_dim)
    else:
        model = GraphTransformer(in_size=input_size,
                                 nb_class=nb_class,
                                 d_model=args.dim_hidden,
                                 dim_feedforward=2*args.dim_hidden,
                                 dropout=args.dropout,
                                 nb_heads=args.nb_heads,
                                 nb_layers=args.nb_layers,
                                 lap_pos_enc=True,
                                 lap_pos_enc_dim=gckn_dim)
    if args.use_cuda:
        model.cuda()
    print("Total number of parameters: {}".format(count_parameters(model)))

    if nb_class == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.warmup is None:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    else:
        lr_steps = (args.lr - 1e-6) / args.warmup
        decay_factor = args.lr * args.warmup ** .5
        def lr_scheduler(s):
            if s < args.warmup:
                lr = 1e-6 + s * lr_steps
            else:
                lr = decay_factor * s ** -.5
            return lr

    test_dset = GraphDataset(dset[test_fold_idx], n_tags, degree=True)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dset.collate_fn())
    if pos_encoder is not None:
        test_dset.pe_list = [dset.pe_list[i] for i in test_fold_idx]


    test_dset.lap_pe_list = gckn_pos_enc_values[-len(test_dset):]

    print("Training...")
    best_val_acc = 0
    best_model = None
    best_epoch = 0
    logs = defaultdict(list)
    start_time = timer()
    for epoch in range(args.epochs):
        print("Epoch {}/{}, LR {:.6f}".format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        train_loss = train_epoch(model, train_loader, criterion, optimizer, lr_scheduler, epoch, args.use_cuda)
        val_acc, val_loss = eval_epoch(model, val_loader, criterion, args.use_cuda)
        test_acc, test_loss = eval_epoch(model, test_loader, criterion, args.use_cuda)

        if args.warmup is None:
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
