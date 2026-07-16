import torch
import numpy as np
from numba import njit, prange

@njit
def bfs_shortest_path(adj_list, startVertex):
    n = adj_list.shape[0]
    path = np.full(n, -1)
    dist = np.full(n, -1)
    q = np.full(n, -1)
    q_start, q_end = -1, 0

    dist[startVertex] = 0
    path[startVertex] = startVertex
    q[0] = startVertex

    while q_start != q_end:
        q_start += 1
        vertex = q[q_start]
        adjVertices = adj_list[vertex]
        cur = 0
        while cur < adj_list.shape[1] and adjVertices[cur] != -1:
            val = adjVertices[cur]
            cur += 1
            if dist[val] == -1:
                q_end += 1
                q[q_end] = val
                dist[val] = dist[vertex] + 1
                path[val] = vertex

    return dist, path

@njit
def get_path_indices(path, max_dist, cur_node):
    n = path.shape[0]
    path_ = np.full(n, -1)
    indices = np.empty((4, n * max_dist), dtype=np.int32)
    offset = 0

    for i in range(n):
        if i == cur_node or path[i] == -1:
            continue
        
        path_[0] = i
        j = 1
        k = i
        while path[k] != cur_node:
            path_[j] = path[k]
            j += 1
            k = path[k]
        path_[j] = cur_node
        path_nodes = path_[: j + 1][::-1]
        j = min(max_dist, j)

        indices[0, offset: offset + j] = i
        indices[1, offset: offset + j] = np.arange(j, dtype=np.int32)
        indices[2, offset: offset + j] = path_nodes[:j]
        indices[3, offset: offset + j] = path_nodes[1: j + 1]
        offset += j

    return indices[:, :offset]

@njit(parallel=True)
def bfs_spatial_pos_with_path(adj_matrix, max_dist=20):
    n = adj_matrix.shape[0]
    adj_list = np.full((n, n), -1)

    for i in range(n):
        cur = 0
        for j in range(n):
            if adj_matrix[i, j]:
                adj_list[i, cur] = j
                cur += 1
    spatial_pos = np.full((n, n), 510)
    path_indices = np.empty((n, 5, n * max_dist), dtype=np.int32)
    path_lengths = np.zeros(n, dtype=np.int32)

    for i in prange(n):
        dist, path = bfs_shortest_path(adj_list, i)
        mask = dist != -1
        spatial_pos[i, mask] = dist[mask]

        indices = get_path_indices(path, max_dist, i)
        l = indices.shape[1]
        path_indices[i, 0, :l] = i
        path_indices[i, 1:, :l] = indices
        path_lengths[i] = l

    flat_path_indices = np.empty((5, path_lengths.sum()), dtype=np.int32)
    offset = 0
    for i in range(n):
        l = path_lengths[i]
        flat_path_indices[:, offset: offset + l] = path_indices[i, :, :l]
        offset += l

    return spatial_pos, flat_path_indices

def to_dense_adj(num_nodes, edge_index, edge_attr=None):
    if edge_attr is None:
        adj = torch.zeros((num_nodes, num_nodes),
                          dtype=torch.bool, device=edge_index.device)
        if edge_index.numel() > 0:
            adj[edge_index[0], edge_index[1]] = True
    else:
        adj = torch.zeros((num_nodes, num_nodes, edge_attr.size(1)),
                          dtype=edge_attr.dtype, device=edge_attr.device)
        if edge_index.numel() > 0:
            adj[edge_index[0], edge_index[1]] = edge_attr
    
    return adj

def compute_shortest_paths(graph, max_dist):
    '''
    Compute sortest paths between all nodes in the graph. graph should have edge_index attribute.
    '''
    adj = to_dense_adj(graph.num_nodes, graph.edge_index)
    spatial_pos, path_indices = bfs_spatial_pos_with_path(adj.numpy(), max_dist)
    graph.spatial_pos = torch.from_numpy(spatial_pos).long()
    graph.path_indices = torch.from_numpy(path_indices).long()
