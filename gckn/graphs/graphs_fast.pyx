# encoding: utf-8
# cython: linetrace=True
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3

from libc.stdlib cimport malloc, free
cimport numpy as np
import numpy as np
np.import_array()


ctypedef np.uint16_t uint16_t
ctypedef np.int64_t int64_t

cdef struct stack:
    int max_size
    int top
    uint16_t *items

cdef stack* init_stack(int capacity) nogil:
    cdef stack *pt = <stack *>malloc(sizeof(stack))
    pt.max_size = capacity
    pt.top = -1
    pt.items = <uint16_t *>malloc(capacity * sizeof(uint16_t))
    return pt

cdef int size(stack *pt) nogil:
    return pt.top + 1

cdef void free_stack(stack *pt) nogil:
    free(pt.items)
    free(pt)

cdef void clear_stack(stack *pt) nogil:
    pt.top = -1

cdef bint empty(stack *pt) nogil:
    return pt.top == -1

cdef uint16_t pop_back(stack *pt) nogil:
    pt.top -= 1
    return pt.items[pt.top + 1]

cdef void push_back(stack *pt, uint16_t x) nogil:
    pt.top += 1
    pt.items[pt.top] = x

cdef class Graph(object):
    cdef int n, m
    cdef uint16_t *edges
    cdef uint16_t **neighbors

    def __cinit__(self, adj_list):
        n = len(adj_list)
        m = sum([len(adj) for adj in adj_list])
        cdef uint16_t *edges = <uint16_t *>malloc(m * sizeof(uint16_t))
        cdef uint16_t **neighbors = <uint16_t **>malloc((n + 1) * sizeof(uint16_t *))
        cdef uint16_t *edges_iter = &edges[0]
        for i in range(n):
            neighbors[i] = &edges_iter[0]
            for adj in adj_list[i]:
                edges_iter[0] = adj
                edges_iter += 1
        neighbors[n] = &edges_iter[0]

        self.n = n
        self.m = self.m
        self.neighbors = neighbors
        self.edges = edges

    def __dealloc__(self):
        if self.edges is not NULL:
            free(self.edges)
        if self.neighbors is not NULL:
            free(self.neighbors)

    cdef void dfs_search(self, uint16_t node, bint *visited) nogil:
        visited[node] = True
        cdef uint16_t *adj = self.neighbors[node]
        while adj != self.neighbors[node + 1]:
            if not visited[adj[0]]:
                self.dfs_search(adj[0], visited)
            adj += 1

    cdef void all_paths(self, uint16_t node, bint *visited, stack *vs,
        int depth, uint16_t[:, :, ::1] all_paths, int64_t[:] counts) nogil:
        visited[node] = True
        push_back(vs, node)

        cdef int j
        cdef int index = size(vs) - 1
        for j in range(size(vs)):
            all_paths[index][counts[index]][j] = vs.items[j]
        counts[index] += 1

        if depth == 0:
            visited[pop_back(vs)] = False
            return

        cdef uint16_t *adj = self.neighbors[node]
        while adj < self.neighbors[node + 1]:
            if not visited[adj[0]]:
                self.all_paths(adj[0], visited, vs, depth - 1, all_paths, counts)
            adj += 1
            if adj == self.neighbors[node + 1]:
                visited[pop_back(vs)] = False

    cdef void all_walks(self, uint16_t node, stack *vs,
        int depth, uint16_t[:, :, ::1] all_paths, int64_t[:] counts) nogil:
        push_back(vs, node)

        cdef int j
        cdef int index = size(vs) - 1
        for j in range(size(vs)):
            all_paths[index][counts[index]][j] = vs.items[j]
        counts[index] += 1

        if depth == 0:
            pop_back(vs)
            return

        cdef uint16_t *adj = self.neighbors[node]
        while adj < self.neighbors[node + 1]:
            self.all_walks(adj[0], vs, depth - 1, all_paths, counts)
            adj += 1
            if adj == self.neighbors[node + 1]:
                pop_back(vs)

def get_paths(graph, int k):
    cdef int n = len(graph.neighbors)
    cdef Graph g = Graph(graph.neighbors)
    cdef int d = graph.mean_neighbor#, graph.max_neighbor)
    cdef int dmax = graph.max_neighbor
    cdef bint *visited = <bint *>malloc(n * sizeof(bint))
    cdef stack *vs = init_stack(k)

    cdef int64_t[:] all_counts = np.zeros(k, dtype=np.int64)
    cdef int64_t[:, ::1] counts = np.empty((n, k), dtype=np.int64)
    cdef int64_t[:] prev_counts = np.zeros(k, dtype=np.int64)

    cdef int i, j
    cdef int max_size = n * d ** (k - 1) * 10

    cdef uint16_t[:, :, ::1] all_paths = np.zeros((k, max_size, k), dtype=np.uint16)

    with nogil:
        for i in range(n):
            for j in range(n):
                visited[j] = False
            clear_stack(vs)
            g.all_paths(i, visited, vs, k - 1, all_paths, all_counts)
            for j in range(k):  
                counts[i, j] = all_counts[j] - prev_counts[j]
                prev_counts[j] = all_counts[j]

    free(visited)
    free_stack(vs)

    paths = []
    for j in range(k):
        paths.append(np.asarray(all_paths[j][:all_counts[j], :j+1]).astype('int64'))

    return paths, np.asarray(counts)

def get_walks(graph, int k):
    cdef int n = len(graph.neighbors)
    cdef Graph g = Graph(graph.neighbors)
    cdef int d = graph.mean_neighbor#, graph.max_neighbor)
    cdef int dmax = graph.max_neighbor
    cdef stack *vs = init_stack(k)

    cdef int64_t[:] all_counts = np.zeros(k, dtype=np.int64)
    cdef int64_t[:, ::1] counts = np.empty((n, k), dtype=np.int64)
    cdef int64_t[:] prev_counts = np.zeros(k, dtype=np.int64)

    cdef int i, j
    cdef int max_size = n * d ** (k - 1) * (dmax)

    cdef uint16_t[:, :, ::1] all_paths = np.zeros((k, max_size, k), dtype=np.uint16)

    with nogil:
        for i in range(n):
            clear_stack(vs)
            g.all_walks(i, vs, k - 1, all_paths, all_counts)
            for j in range(k):  
                counts[i, j] = all_counts[j] - prev_counts[j]
                prev_counts[j] = all_counts[j]

    free_stack(vs)

    paths = []
    for j in range(k):
        paths.append(np.asarray(all_paths[j][:all_counts[j], :j+1]).astype('int64'))

    return paths, np.asarray(counts)
