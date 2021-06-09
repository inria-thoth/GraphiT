import os
import pickle
import torch
import torch.nn.functional as F
import torch_geometric.utils as utils
from gckn.data import PathLoader, S2VGraph
from gckn.models import PathSequential


def compute_gckn_pe(train_graphs, test_graphs=None, path_size=3, hidden_size=32,
                    batch_size=64, sigma=0.5, pooling='mean',
                    aggregation=True, normalize=False, use_cuda=False):
    data_loader = PathLoader(train_graphs, path_size, batch_size, True)
    input_size = data_loader.input_size
    data_loader.get_all_paths()

    model = PathSequential(input_size, [hidden_size], [path_size],
        kernel_args_list=[sigma], pooling=pooling, aggregation=aggregation)

    model.unsup_train(data_loader, n_sampling_paths=300000, use_cuda=use_cuda)

    scaler = None
    if normalize:
        from sklearn.preprocessing import StandardScaler
        train_pe = model.encode(data_loader, use_cuda=use_cuda)
        scaler = StandardScaler()
        scaler.fit(torch.cat(train_pe, dim=0).numpy())
        train_pe = [torch.from_numpy(
            scaler.transform(pe.numpy())) for pe in train_pe]
    else:
        train_pe = model.encode(data_loader, use_cuda=use_cuda)

    if test_graphs is not None:
        data_loader = PathLoader(test_graphs, path_size, batch_size, True)
        data_loader.get_all_paths()
        test_pe = model.encode(data_loader, use_cuda=use_cuda)
        if scaler is not None:
            test_pe = [torch.from_numpy(
                scaler.transform(pe.numpy())) for pe in test_pe]
        return train_pe + test_pe
    return train_pe, None


def get_adj_list(g):
    neighbors = [[] for _ in range(g.num_nodes)]
    for k in range(g.edge_index.shape[-1]):
        i, j = g.edge_index[:, k]
        neighbors[i.item()].append(j.item())
    return neighbors

def convert_dataset(dataset, n_tags=None):
    """Convert a pytorch geometric dataset to gckn dataset
    """
    if dataset is None:
        return dataset
    graph_list = []
    for i, g in enumerate(dataset):
        new_g = S2VGraph(g, g.y)
        new_g.neighbors = get_adj_list(g)
        if n_tags is not None:
            new_g.node_features = F.one_hot(g.x.view(-1).long(), n_tags).numpy()
        else:
            new_g.node_features = g.x.numpy()
        degree_list = utils.degree(g.edge_index[0], g.num_nodes).numpy()
        new_g.max_neighbor = max(degree_list)
        new_g.mean_neighbor = (sum(degree_list) + len(degree_list) - 1) // len(degree_list)
        graph_list.append(new_g)
    return graph_list


class GCKNEncoding(object):
    def __init__(self, savepath, dim, path_size, sigma=0.6, pooling='sum', aggregation=True,
                 normalize=True):
        self.savepath = savepath
        self.dim = dim
        self.path_size = path_size
        self.sigma = sigma
        self.pooling = pooling
        self.aggregation = aggregation
        self.normalize = normalize

        self.pos_enc_dim = dim
        if aggregation:
            self.pos_enc_dim = path_size * dim

    def apply_to(self, train_dset, test_dset=None, batch_size=64, n_tags=None):
        """take pytorch geometric dataest as input
        """
        saved_pos_enc = self.load()
        if saved_pos_enc is not None:
            dset_len = len(train_dset) if test_dset is None else len(train_dset) + len(test_dset)
            if len(saved_pos_enc) != dset_len:
                raise ValueError("Incorrect save path!")
            return saved_pos_enc
        train_dset = convert_dataset(train_dset, n_tags)
        test_dset = convert_dataset(test_dset, n_tags)
        pos_enc = compute_gckn_pe(
            train_dset, test_dset, path_size=self.path_size, hidden_size=self.dim,
            batch_size=batch_size, sigma=self.sigma, pooling=self.pooling,
            aggregation=self.aggregation, normalize=self.normalize)
        self.save(pos_enc)
        return pos_enc

    def save(self, pos_enc):
        if self.savepath is None:
            return
        if not os.path.isfile(self.savepath):
            with open(self.savepath, 'wb') as handle:
                pickle.dump(pos_enc, handle)

    def load(self):
        if not os.path.isfile(self.savepath):
            return None
        with open(self.savepath, 'rb') as handle:
            pos_enc = pickle.load(handle)
        return pos_enc


if __name__ == "__main__":
    from torch_geometric import datasets
    data_path = '../dataset/ZINC'
    n_tags = 28
    dset = datasets.ZINC(data_path, subset=True, split='val')
    dset = convert_dataset(dset, 28)
    from gckn.graphs import get_paths
    graphs_pe = compute_gckn_pe(dset, batch_size=64, aggregation=True)
    print(len(graphs_pe))
    print(graphs_pe[0])
    print(graphs_pe[0].shape)


