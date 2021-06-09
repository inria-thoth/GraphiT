import os
import re
import statistics
import numpy as np
import networkx as nx


class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy float tensor, one-hot representation of the tag that is used as input to neural nets
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0

        self.max_neighbor = 0
        self.mean_neighbor = 0


def load_graphdata(dataname, datadir='dataset', max_nodes=None, edge_labels=False):
    """ Read data from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        graph index starts with 1 in file

    Returns:
        List of networkx objects with graph and node labels
    """
    prefix = os.path.join(datadir, dataname, dataname)
    filename_graph_indic = prefix + "_graph_indicator.txt"
    # index of graphs that a given node belongs to
    graph_indic = {}
    with open(filename_graph_indic) as f:
        i = 1
        for line in f:
            line = line.strip("\n")
            graph_indic[i] = int(line)
            i += 1

    filename_nodes = prefix + "_node_labels.txt"
    node_labels = []
    min_label_val = None
    try:
        with open(filename_nodes) as f:
            for line in f:
                line = line.strip("\n")
                l = int(line)
                node_labels += [l]
                if min_label_val is None or min_label_val > l:
                    min_label_val = l
        # assume that node labels are consecutive
        num_unique_node_labels = max(node_labels) - min_label_val + 1
        node_labels = [l - min_label_val for l in node_labels]
    except IOError:
        print("No node labels")

    filename_node_attrs = prefix + "_node_attributes.txt"
    node_attrs = []
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [
                    float(attr) for attr in re.split("[,\s]+", line) if not attr == ""
                ]
                node_attrs.append(np.array(attrs))
    except IOError:
        print("No node attributes")

    label_has_zero = False
    filename_graphs = prefix + "_graph_labels.txt"
    graph_labels = []

    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line = line.strip("\n")
            val = int(line)
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)

    label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])

    if edge_labels:
        # For Tox21_AHR we want to know edge labels
        filename_edges = prefix + "_edge_labels.txt"
        edge_labels = []

        edge_label_vals = []
        with open(filename_edges) as f:
            for line in f:
                line = line.strip("\n")
                val = int(line)
                if val not in edge_label_vals:
                    edge_label_vals.append(val)
                edge_labels.append(val)

        edge_label_map_to_int = {val: i for i, val in enumerate(edge_label_vals)}

    filename_adj = prefix + "_A.txt"
    adj_list = {i: [] for i in range(1, len(graph_labels) + 1)}
    # edge_label_list={i:[] for i in range(1,len(graph_labels)+1)}
    index_graph = {i: [] for i in range(1, len(graph_labels) + 1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line = line.strip("\n").split(",")
            e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0, e1))
            index_graph[graph_indic[e0]] += [e0, e1]
            # edge_label_list[graph_indic[e0]].append(edge_labels[num_edges])
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k] = [u - 1 for u in set(index_graph[k])]

    graphs = []
    for i in range(1, 1 + len(adj_list)):
        # indexed from 1 here
        G = nx.from_edgelist(adj_list[i])

        if max_nodes is not None and G.number_of_nodes() > max_nodes:
            continue

        # add features and labels
        G.graph["label"] = graph_labels[i - 1]

        # Special label for aromaticity experiment
        # aromatic_edge = 2
        # G.graph['aromatic'] = aromatic_edge in edge_label_list[i]

        for u in G.nodes():
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u - 1]
                node_label_one_hot[node_label] = 1
                G.nodes[u]["label"] = node_label_one_hot
                G.nodes[u]["tag"] = node_label
            if len(node_attrs) > 0:
                G.nodes[u]["feat"] = node_attrs[u - 1]
        if len(node_attrs) > 0:
            G.graph["feat_dim"] = node_attrs[0].shape[0]

        # relabeling
        mapping = {}
        it = 0
        if float(nx.__version__) < 2.0:
            for n in G.nodes():
                mapping[n] = it
                it += 1
        else:
            for n in G.nodes:
                mapping[n] = it
                it += 1

        # indexed from 0
        G = nx.relabel_nodes(G, mapping)

        l = int(G.graph['label'])
        adj = [[] for i in range(len(G))]
        for i, j in G.edges():
            adj[i].append(j)
            adj[j].append(i)

        degree_list = []
        for i in range(len(G)):
            degree_list.append(len(adj[i]))
        if len(node_attrs) > 0:
            node_features = [G.nodes[u]['feat'] for u in G.nodes()]
            node_features = np.asarray(node_features)
            if len(node_labels) > 0:
                node_labels_one_hot = np.asarray([G.nodes[u]['label'] for u in G.nodes()])
                #node_features = np.hstack([node_features, node_labels_one_hot])
        else:
            node_features = [G.nodes[u]['label'] for u in G.nodes()]
            node_features = np.asarray(node_features)
            node_labels_one_hot = node_features
        # node_features = node_features / np.linalg.norm(node_features, axis=-1, keepdims=True).clip(min=1e-05)
        # print(node_labels)
        G = S2VGraph(G, l, node_labels)
        if edge_labels:
            G.edge_labels = edge_labels
        G.neighbors = adj
        G.max_neighbor = max(degree_list)
        G.mean_neighbor = (sum(degree_list) + len(degree_list) - 1) // len(degree_list)
        G.degree_list = degree_list
        G.node_features = node_features
        G.node_labels = node_labels_one_hot
        graphs.append(G)
    return graphs, int(max(graph_labels)) + 1

def get_motif(mask, path_indices, graph, max_component=True, eps=0.1):
    if not isinstance(mask, list):
        mask = [mask]
    if not isinstance(path_indices, list):
        path_indices = [path_indices]
    g = nx.Graph()
    g.add_nodes_from(graph.nodes())
    n = len(g.nodes())
    for node in graph.nodes():
        g.nodes[node]['tag'] = graph.nodes[node]['tag']

    # edge_list = []
    adj = np.zeros((n, n))
    for m, path in zip(mask, path_indices):
        if len(path[0]) <= 1:
            continue
        for i in range(len(m)):
            if m[i] > eps:
                p = path[i]
                for j in range(len(p) - 1):
                    adj[p[j], p[j+1]] += m[i]
    adj /= np.max(adj)
    edge_list = [(i, j, adj[i, j]) for i in range(n) for j in range(n) if adj[i, j] > eps]
    # print(adj)
    g.add_weighted_edges_from(edge_list)

    if max_component:
        largest_cc = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_cc).copy()
    else:
        # remove zero degree nodes
        g.remove_nodes_from(list(nx.isolates(g)))
    return g


def log_graph(
    graph,
    outdir,
    filename,
    identify_self=False,
    nodecolor="tag",
    fig_size=(4, 3),
    dpi=300,
    label_node_feat=True,
    edge_vmax=None,
    args=None,
    eps=1e-6,
):
    """
    Args:
        nodecolor: the color of node, can be determined by 'label', or 'feat'. For feat, it needs to
            be one-hot'
    """
    if len(graph.edges) == 0:
        return
    import matplotlib.pyplot as plt
    plt.switch_backend("agg")
    cmap = plt.get_cmap("tab20")
    plt.switch_backend("agg")
    fig = plt.figure(figsize=fig_size, dpi=dpi)

    node_colors = []
    # edge_colors = [min(max(w, 0.0), 1.0) for (u,v,w) in Gc.edges.data('weight', default=1)]
    edge_colors = [w for (u, v, w) in graph.edges.data("weight", default=1)]

    # maximum value for node color
    vmax = 19
    # for i in graph.nodes():
    #     if nodecolor == "feat" and "feat" in graph.nodes[i]:
    #         num_classes = graph.nodes[i]["feat"].size()[0]
    #         if num_classes >= 10:
    #             cmap = plt.get_cmap("tab20")
    #             vmax = 19
    #         elif num_classes >= 8:
    #             cmap = plt.get_cmap("tab10")
    #             vmax = 9
    #         break

    feat_labels = {}
    for i in graph.nodes():
        if identify_self and "self" in graph.nodes[i]:
            node_colors.append(0)
        elif nodecolor == "tag" and "tag" in graph.nodes[i]:
            node_colors.append(graph.nodes[i]["tag"])
            feat_labels[i] = graph.nodes[i]["tag"]
        elif nodecolor == "feat" and "feat" in graph.nodes[i]:
            # print(Gc.nodes[i]['feat'])
            feat = graph.nodes[i]["feat"].detach().numpy()
            # idx with pos val in 1D array
            feat_class = 0
            for j in range(len(feat)):
                if feat[j] == 1:
                    feat_class = j
                    break
            node_colors.append(feat_class)
            feat_labels[i] = feat_class
        else:
            node_colors.append(1)
    if not label_node_feat:
        feat_labels = None

    plt.switch_backend("agg")
    fig = plt.figure(figsize=fig_size, dpi=dpi)

    if graph.number_of_nodes() == 0:
        raise Exception("empty graph")
    if graph.number_of_edges() == 0:
        raise Exception("empty edge")
    # remove_nodes = []
    if len(graph.nodes) > 20:
        pos_layout = nx.kamada_kawai_layout(graph, weight=None)
        # pos_layout = nx.spring_layout(graph, weight=None)
    else:
        pos_layout = nx.kamada_kawai_layout(graph, weight=None)

    weights = [d for (u, v, d) in graph.edges(data="weight", default=1)]
    if edge_vmax is None:
        edge_vmax = statistics.median_high(
            [d for (u, v, d) in graph.edges(data="weight", default=1)]
        )
    min_color = min([d for (u, v, d) in graph.edges(data="weight", default=1)])
    # color range: gray to black
    edge_vmin = 2 * min_color - edge_vmax
    print(edge_vmin)
    print(edge_vmax)
    print(edge_colors)
    nx.draw(
        graph,
        pos=pos_layout,
        with_labels=False,
        font_size=4,
        labels=feat_labels,
        node_color=node_colors,
        vmin=0,
        vmax=vmax,
        cmap=cmap,
        edge_color=edge_colors,
        edge_cmap=plt.get_cmap("Greys"),
        edge_vmin=edge_vmin-eps,
        edge_vmax=edge_vmax,
        width=1.3,
        node_size=100,
        alpha=0.9,
        arrows=False
    )
    fig.axes[0].xaxis.set_visible(False)
    fig.canvas.draw()

    save_path = os.path.join(outdir, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    nx.write_gpickle(graph, os.path.splitext(save_path)[0] + '.gpickle')
    plt.savefig(save_path, format="pdf")


if __name__ == "__main__":
    graphs = load_data('Mutagenicity', '../dataset')
    # print(list(graphs[0].adjacency()))
    print([list(adj.keys()) for _, adj in graphs[0].adjacency()])
    print(graphs[0].graph['label'])
    print(len(graphs))
