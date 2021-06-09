# -*- coding: utf-8 -*-
import numpy as np


def exp(x, sigma=0.6):
    return np.exp(1./sigma ** 2 * (x - 1.))
    # return (x != 0) * np.exp(1./sigma ** 2 * (x - 1.))


def dfs(graph, filters, cum=False):
    """
    graph.node_features: n x d
    filters: k x p x d
    output: k x p
    """
    n = len(graph.neighbors)
    print(n)
    k, p, d = filters.shape
    # base_sim: n x k x p
    # base_sim = graph.node_features.dot(filters.T)
    base_sim = np.tensordot(graph.node_features, filters, axes=[[-1], [-1]])
    output = np.zeros((k, p))
    current_out = np.zeros((k, p))
    # previous_out = np.zeros((k, p))
    divider = np.arange(1, k + 1).reshape(-1, 1).astype(float)
    count = [0]

    def dfs_iterative(node, depth, visited, previous_out):
        current_out[k - 1 - depth:] += base_sim[node, k - 1 - depth, :]
        visited.add(node)
        if depth == 0:
            # current_out[k - 1] += base_sim[node, k - 1, :]
            # print('--')
            # print(current_out)
            # print('--')
            output[:] += exp(current_out / divider)
            # print(current_out / divider)
            # current_out[k - 1 - depth:] -= base_sim[node, k - 1 - depth, :]
            count[0] += 1
            return
        for adj in graph.neighbors[node]:
            if adj not in visited:
                # previous_out[:k - 1 - depth] = 0
                # previous_out = np.zeros((k, p))
                # previous_out[k - depth:] = current_out[k - depth:]
                previous_out = current_out.copy()
                previous_out[:k - depth] = 0
                prev_visited = visited.copy()
                # print('c')
                # print(current_out)
                # print('c')
                dfs_iterative(adj, depth - 1, visited, previous_out)
                # current_out[:k - depth] = 0 #previous_out[:]
                current_out[:] = previous_out
                visited.clear()
                visited.update(prev_visited)

    for i in range(n):
        visited = set()
        current_out[:] = 0
        # previous_out[:] = 0
        dfs_iterative(i, k - 1, visited, previous_out=None)
        # print(i)
        # print(visited)
        # break
    # if cum:
    #     output = output.sum(0)
    # else:
    #     output = output[-1]
    # print(count)
    return output

# def bfs(graph, filters, cum=False):
#     n = len(graph.neighbors)
#     print(n)
#     k, p, d = filters.shape
#     # base_sim: n x k x p
#     base_sim = np.tensordot(graph.node_features, filters, axes=[[-1], [-1]])
#     output = np.zeros((k, p))
#     current_out = np.zeros((k, p))
#     divider = np.arange(1, k + 1).reshape(-1, 1).astype(float)
#     count = [0]

#     visited = set()

#     for i in range(n):
#         queue = [i]
#         depth = [0]
#         # counters = 0
#         # depths = [0]
#         while len(queue) > 0 and depth.pop(0) < k:
#             node = queue.pop(0)
#             if node not in visited():
#                 visited.add(node)
#                 depth = depths[node]
#                 neighbors = graph.neighbors[node] - visited
#                 queue.extend(neighbors)
#                 # depths[neighbors] = depth + 1
#                 depth.extend([depth + 1] * len(neighbors))


def pathkernel_compare(graph1, graph2, k, normalize=False, cum=False):
    """
    graph1.node_features: n1 x d
    graph2.node_features: n2 x d
    output: k x 1
    """
    n1 = len(graph1.neighbors)
    n2 = len(graph2.neighbors)
    # base_sim: n1 x n2
    base_sim = graph1.node_features.dot(graph2.node_features.T)
    current_out = np.zeros(k)
    output = np.zeros(k)
    divider = np.arange(1, k + 1)
    count = np.zeros(k)
    # incr = 1. / np.arange(1, k + 1)
    # print(incr)
    print(base_sim.shape)
    print(base_sim.sum())

    def dfs_iterative(node1, node2, depth, visited1, visited2, previous_out):
        current_out[k - 1 - depth:] += base_sim[node1, node2]
        count[k - 1 - depth] += 1  # incr[k - 1 - depth:]
        visited1.add(node1)
        visited2.add(node2)
        # visited1.append(node1)
        # visited2.append(node2)

        is_leaf = (len(set(graph1.neighbors[node1]) - visited1) == 0) or (
                len(set(graph2.neighbors[node2]) - visited2) == 0)
        # print(visited1 - set(graph1.neighbors[node1]))
        if depth == 0 or is_leaf:
            current_out[k - depth:] = 0
            output[:] += exp(current_out / divider)
            # output[:] += current_out / divider
            # print('out')
            # print(current_out)
            return

        for adj1 in graph1.neighbors[node1]:
            if adj1 not in visited1:
                prev = visited2.copy()
                prev1 = visited1.copy()
                for adj2 in graph2.neighbors[node2]:
                    # visited1 = prev1
                    if adj2 not in visited2:
                        # print('in')
                        # print(visited1)
                        # print(prev1)
                        visited1 = prev1
                        previous_out = np.zeros(k)
                        previous_out[k - depth:] = current_out[k - depth:]
                        dfs_iterative(adj1, adj2, depth - 1, visited1,
                                      visited2, previous_out)
                        current_out[:] = previous_out
                        # print('h')
                        # print(visited2)
                # visited2[:] = prev
                visited2.clear()
                visited2.update(prev)

    for i1 in range(n1):
        for i2 in range(n2):
            visited1 = set()
            visited2 = set()
            # i1 = 0
            # i2 = 0
            # visited1 = []
            # visited2 = []
            current_out[:] = 0
            # count[:] = 0
            dfs_iterative(i1, i2, k - 1, visited1, visited2, previous_out=None)
            # print(count)
            # print('path')
            # print(visited1)
            # print(visited2)
            # print(count)
            # print('path')
            # # break
            # return
        #     break
        # break
    # count[1:] /= 4
    # output[1:] /=4
    print(count)

    if normalize:
        output /= count

    # if cum:
    #     output = output.sum()
    # else:
    #     output = output[-1]
    return output


def get_paths(graph, k):
    n = len(graph.neighbors)
    current_out = np.zeros(k, dtype=int)
    all_paths = [set() for i in range(k)]
    out_paths = []
    # i = 0

    def dfs_iterative(node, depth, visited, previous_out):
        current_out[k - 1 - depth] = node
        visited.add(node)
        # visited.append(node)
        all_paths[k - 1 - depth].add(tuple(current_out[:k - depth]))

        if depth == 0:
            # all_paths[-1].add(current_out.copy())
            return

        for adj in graph.neighbors[node]:
            if adj not in visited:
                prev = visited.copy()
                previous_out = current_out.copy()
                dfs_iterative(adj, depth - 1, visited, previous_out)
                current_out[:] = previous_out
                # visited[:] = prev
                visited.clear()
                visited.update(prev)

    for i in range(n):
        # all_paths = []
        visited = set()
        # visited = []
        dfs_iterative(i, k - 1, visited, previous_out=None)
        # print(all_paths[0])
        # print(visited)
        # break
        # out_paths.append(np.asarray(all_paths))
    # print(all_paths[0])
    # print(all_paths[1])
    return [np.asarray(list(p)) for p in all_paths]


def pathkernel_compare2(graph1, graph2, k, normalize=False, cum=False):
    """
    graph1.node_features: n1 x d
    graph2.node_features: n2 x d
    output: k x 1
    """
    n1 = len(graph1.neighbors)
    n2 = len(graph2.neighbors)
    # base_sim: n1 x n2
    base_sim = graph1.node_features.dot(graph2.node_features.T)
    # current_out = np.zeros(k)
    # output = np.zeros(k)
    # divider = np.arange(1, k + 1)
    # count = np.zeros(k)
    # incr = 1. / np.arange(1, k + 1)
    # print(incr)
    print(base_sim.shape)
    print(base_sim.sum())

    path1 = get_paths(graph1, k)
    path2 = get_paths(graph2, k)
    # print(path1[1])
    # print(len(path1))
    # out = 0
    # out = [0 for i in range(k)]
    out = [0] * k
    for i in range(k):
        for j in range(k):
            if j >= i:
                out[j] += base_sim[np.ix_(path1[j][:, i].reshape(-1),
                                   path2[j][:, i].reshape(-1))] / (j + 1.)
    # out /= k
    # print(out.shape)
    return np.asarray([exp(o).mean() for o in out])
    # return exp(out).sum()#/4.


def pathkernel(graphs, k, coef=2.0, normalize=False, cum=False, sigma=0.6):
    all_paths = [get_paths(g, k) for g in graphs]
    print('all paths found!')
    n = len(graphs)
    gram = np.zeros((n, n))
    coef = coef ** np.arange(k)
    print(coef)

    def kernel_value(base_sim, path1, path2):
        out = [0] * k
        for i in range(k):
            for j in range(k):
                if j >= i:
                    if path1[j].size == 0 or path2[j].size == 0:
                        continue
                    out[j] += base_sim[np.ix_(path1[j][:, i].reshape(-1),
                                              path2[j][:, i].reshape(-1))
                                       ] / (j + 1.)
        if normalize:
            out = np.asarray([exp(o, sigma).mean() for o in out])
        else:
            out = np.asarray([exp(o, sigma).sum() for o in out])
        if cum:
            return (out * coef).sum()
        else:
            return out[-1]

    for i in range(n):
        # print(graphs[i].node_features)
        for j in range(n):
            if j >= i:
                base_sim = graphs[i].node_features.dot(
                        graphs[j].node_features.T)
                val = kernel_value(base_sim, all_paths[i], all_paths[j])
                gram[i, j] = val
                gram[j, i] = val
    return gram


if __name__ == "__main__":
    import argparse
    from gckn.data import load_data
    parser = argparse.ArgumentParser(description='path kernel computation')
    parser.add_argument('--dataset', type=str, default="PTC",
                        help='name of dataset (default: PTC)')
    parser.add_argument('--size', type=int, default=2)
    parser.add_argument('--normalize', action='store_true', help='take mean')
    parser.add_argument('--sigma', type=float, default=0.6)
    parser.add_argument('--coef', type=float, default=1.0)
    args = parser.parse_args()

    np.random.seed(0)
    graphs, _ = load_data(args.dataset, '../dataset', True)

    # test dfs
    # g = graphs[0]
    # d = g.node_features.shape[-1]
    # # filters = np.ones((3, 2, d))
    # filters = np.random.rand(3, 2, d)
    # print(g.neighbors)
    # out = dfs(g, filters)
    # print(out)

    # pathkernel
    # g1 = graphs[0]
    # g2 = graphs[0]
    # print(len(g1.neighbors))
    # print(len(g2.neighbors))
    # print(g1.neighbors)
    # print(g2.neighbors)

    # out = pathkernel_compare2(g1, g2, 10, False, False)
    # print(out)

    # Gram matrix
    print(graphs[151].neighbors)
    out = pathkernel(graphs, args.size, coef=args.coef,
                     normalize=args.normalize, cum=True, sigma=args.sigma)
    print(out)
    labels = [g.label for g in graphs]
    print(labels)
    # np.save('../out/gram', out)
    # np.save('../out/labels', labels)
