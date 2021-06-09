import os
import pickle
import torch
import torch.nn.functional as F
import torch_geometric.utils as utils
from gckn.data import PathLoader, S2VGraph


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
