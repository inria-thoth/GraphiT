# -*- coding: utf-8 -*-
import networkx as nx
import numpy as np
import pickle
import torch
from sklearn.model_selection import StratifiedKFold
from .data_io import load_graphdata
from .graphs import get_paths, get_walks


DATA = ['Mutagenicity', 'BZR', 'COX2', 'ENZYMES', 'PROTEINS_full']


class GraphLoader(object):
    """
    This class takes a list of graphs and transforms it into a
    data_loader.
    """
    def __init__(self, path_size, batch_size, dataset, walk):
        self.path_size = path_size
        self.batch_size = batch_size
        self.dataset = dataset
        self.walk = walk

    def transform(self, graphs):
        data_loader = PathLoader(graphs, max(self.path_size), self.batch_size,
                                 True, dataset=self.dataset, walk=self.walk)
        if self.dataset != 'COLLAB' or max(self.path_size) <= 2:
            data_loader.get_all_paths()
        return data_loader

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy float tensor, one-hot representation of the
            tag that is used as input to neural nets
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.pe = None

        self.max_neighbor = 0
        self.mean_neighbor = 0


def load_data(dataset, datapath='dataset', degree_as_tag=False,
              pos_enc=None, pe_size=None):
    '''
        dataset: name of dataset
    '''

    print('loading data')
    if dataset in DATA:
        return load_graphdata(dataset, datapath)
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('{}/{}/{}.txt'.format(datapath, dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
                node_feature_flag = True
            else:
                node_features = None
                node_feature_flag = False

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)
        g.mean_neighbor = (
                sum(degree_list) + len(degree_list) - 1) // len(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        deg_list = list(dict(g.g.degree(range(len(g.g)))).values())

    if degree_as_tag:
        for g in g_list:
            g.node_tags = list(dict(g.g.degree(range(len(g.g)))).values())

    if pos_enc == 'lappe':
        with open('{}/{}/lappe_list.pkl'.format(
                datapath, dataset), 'rb') as handle:
            lappe_list = pickle.load(handle)
        for g, lappe in zip(g_list, lappe_list):
            g.pe = lappe
    elif pos_enc == 'diffusion':
        with open('{}/{}/diffusion_{}_list.pkl'.format(
                datapath, dataset, pe_size), 'rb') as handle:
            graphwave_list = pickle.load(handle)
        for g, diff_kernel in zip(g_list, graphwave_list):
            g.pe = diff_kernel
    # Extracting unique tag labels
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]: i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = np.zeros([len(g.node_tags), len(tagset)])
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1

    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)


def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list


def get_path_indices(paths, n_paths_for_graph, n_nodes):
    """
    paths: all_paths x k
    n_paths:: n_graphs (sum=all_paths)
    """
    incr_indices = torch.cat([torch.zeros(1, dtype=torch.long), n_nodes[:-1]])
    incr_indices = incr_indices.cumsum(dim=0)
    incr_indices = incr_indices.repeat_interleave(n_paths_for_graph, dim=0).view(-1, 1)
    paths = paths + incr_indices
    return paths


class PathLoader(object):
    def __init__(self, graphs, k, batch_size, aggregation=True, dataset='MUTAG', padding=False, walk=False, mask=False):
        # self.data = data
        self.dataset = dataset
        self.graphs = graphs
        self.batch_size = batch_size
        self.aggregation = aggregation
        self.input_size = graphs[0].node_features.shape[-1]
        self.n = len(graphs)
        self.k = k
        self.data = None
        self.labels = None
        self.padding = padding
        self.walk = walk
        self.mask = mask

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def get_all_paths(self, dirname=None):
        all_paths = []
        n_paths = []
        n_nodes = torch.zeros(self.n, dtype=torch.long)
        if self.aggregation:
            n_paths_for_graph = torch.zeros((self.n, self.k), dtype=torch.long)
        else:
            n_paths_for_graph = torch.zeros(self.n, dtype=torch.long)
        features = []
        labels = torch.zeros(self.n, dtype=torch.long)
        mask_vec = []
        if dirname is not None and self.dataset == 'COLLA':
            try:
                self.data = torch.load(dirname + '/all_paths_{}.pkl'.format(self.k))
                self.labels = self.data['labels']
                return
            except:
                pass
        for i, g in enumerate(self.graphs):
            # print(i)
            if self.walk:
                p, c = get_walks(g, self.k)
            else:
                p, c = get_paths(g, self.k)
            if self.aggregation:
                all_paths.append([torch.from_numpy(p[j]) for j in range(self.k)])
                n_paths.append([torch.from_numpy(c[:, j]) for j in range(self.k)])
                n_paths_for_graph[i] = torch.LongTensor([len(p[j]) for j in range(self.k)])
                if self.mask:
                    mask_vec.append([torch.ones(len(p[j])) for j in range(self.k)])
            else:
                all_paths.append(torch.from_numpy(p[-1]))
                n_paths.append(torch.from_numpy(c[:, -1]))
                n_paths_for_graph[i] = len(p[-1])
                mask_vec.append(torch.ones(len(p[-1])))
            n_nodes[i] = len(g.neighbors)
            features.append(torch.from_numpy(g.node_features.astype('float32')))

            labels[i] = g.label

        self.data = {
            'features': features,
            'paths': all_paths,
            'n_paths': n_paths,
            'n_paths_for_graph': n_paths_for_graph,
            'n_nodes': n_nodes,
            'labels': labels
        }
        self.mask_vec = mask_vec
        self.labels = labels
        if dirname is not None and self.dataset == 'COLLA':
            torch.save(self.data, dirname + '/all_paths_{}.pkl'.format(self.k))

    def make_batch(self, shuffle=True):
        if self.data is None:
            # raise ValueError('Plase first run self.get_all_paths() to compute paths!')
            if self.labels is None:
                self.labels = torch.LongTensor([g.label for g in self.graphs])
            if shuffle:
                indices = np.random.permutation(self.n)
            else:
                indices = list(range(self.n))
            for index in range(0, self.n, self.batch_size):
                idx = indices[index:min(index + self.batch_size, self.n)]
                size = len(idx)
                # current_features = torch.cat([torch.from_numpy(
                #     self.graphs[i].node_features.astype('float32')) for i in idx])
                current_features = []
                current_n_nodes = torch.zeros(size, dtype=torch.long)
                if self.aggregation:
                    current_paths = [[] for i in range(self.k)]
                    current_n_paths = [[] for i in range(self.k)]
                    current_n_paths_for_graph = torch.zeros((size, self.k), dtype=torch.long)
                else:
                    current_paths = []
                    current_n_paths = []
                    current_n_paths_for_graph = torch.zeros(size, dtype=torch.long)

                for i, g_index in enumerate(idx):
                    g = self.graphs[g_index]
                    current_features.append(torch.from_numpy(
                        g.node_features.astype('float32')))
                    if self.walk:
                        p, c = get_walks(g, self.k)
                    else:
                        p, c = get_paths(g, self.k)
                    current_n_nodes[i] = len(g.neighbors)
                    if self.aggregation:
                        for j in range(self.k):
                            current_paths[j].append(torch.from_numpy(p[j]))
                            current_n_paths[j].append(torch.from_numpy(c[:, j]))
                            current_n_paths_for_graph[i, j] = len(p[j])
                    else:
                        current_paths.append(torch.from_numpy(p[-1]))
                        current_n_paths.append(torch.from_numpy(c[:, -1]))
                        current_n_paths_for_graph[i] = len(p[-1])
                current_features = torch.cat(current_features)
                if self.aggregation:
                    for j in range(self.k):
                        current_paths[j] = get_path_indices(
                            torch.cat(current_paths[j]), current_n_paths_for_graph[:, j], current_n_nodes)
                        current_n_paths[j] = torch.cat(current_n_paths[j])
                else:
                    current_paths = get_path_indices(
                        torch.cat(current_paths), current_n_paths_for_graph, current_n_nodes)
                    current_n_paths = torch.cat(current_n_paths)
                yield {'features': current_features,
                       'paths': current_paths,
                       'n_paths': current_n_paths,
                       'n_nodes': current_n_nodes,
                       'labels': self.labels[idx]}
            return

        if shuffle:
            indices = np.random.permutation(self.n)
        else:
            indices = list(range(self.n))
        features = self.data['features']

        for index in range(0, self.n, self.batch_size):
            idx = indices[index:min(index + self.batch_size, self.n)]
            current_features = torch.cat([features[i] for i in idx])
            if self.padding:
                current_features = torch.cat([torch.zeros(self.input_size).view(1, -1), current_features])

            if self.aggregation:
                current_paths = [torch.cat([self.data['paths'][i][j] for i in idx]) for j in range(self.k)]
            else:
                current_paths = torch.cat([self.data['paths'][i] for i in idx]) + self.padding
            current_n_paths_for_graph = self.data['n_paths_for_graph'][idx]
            current_n_nodes = self.data['n_nodes'][idx]

            if self.aggregation:
                current_paths = [get_path_indices(
                    current_paths[j], current_n_paths_for_graph[:, j],
                    current_n_nodes) for j in range(self.k)]
                current_n_paths = [torch.cat(
                    [self.data['n_paths'][i][j] for i in idx]) for j in range(self.k)]
            else:
                current_paths = get_path_indices(
                    current_paths, current_n_paths_for_graph, current_n_nodes)
                current_n_paths = torch.cat([self.data['n_paths'][i] for i in idx])
            yield {'features': current_features,
                   'paths': current_paths,
                   'n_paths': current_n_paths,
                   'n_nodes': current_n_nodes,
                   'labels': self.labels[idx]}


if __name__ == "__main__":
    np.random.seed(0)
    graphs, _ = load_data('PTC', '../dataset')
    # graphs = [graphs[i] for i in range(2)]
    k = 4
    # out = get_all_paths(graphs, k, True)
    # # print(out['paths'])
    # # print(out['n_paths'])
    # # print(out['n_paths_for_graph'])
    # # print(node_features)
    # for data in make_batch(out, 2, aggregation=True):
    #     print(data['features'].shape)
    #     print([data['paths'][i].shape for i in range(k)])
    #     print([data['n_paths'][i].sum() for i in range(k)])
    #     # print(data['paths'].shape)
    #     # print(data['n_paths'].shape)
    #     print(data['n_nodes'])
    #     break
    data_loader = PathLoader(graphs, k, 16, aggregation=True, padding=True)
    data_loader.get_all_paths()
    for data in data_loader.make_batch():
        print(data['features'].shape)
        print([data['paths'][i].shape for i in range(k)])
        # print(data['features'])
        # print(data['paths'])#[-1])
        print([data['n_paths'][i].sum() for i in range(k)])
        # print(data['paths'].shape)
        # print(data['n_paths'].shape)
        print(data['n_nodes'])
        break
