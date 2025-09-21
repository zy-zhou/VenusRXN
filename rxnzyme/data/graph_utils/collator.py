import torch
import torch.nn.functional as F

def pad_1d(x, padlen, value=0): # -> [pad_len]
    xlen = x.size(0)
    if xlen >= padlen:
        return x
    padding = padlen - xlen
    return F.pad(x, (0, padding), value=value)

def pad_2d(x, padlen, value=0): # -> [padlen, xdim]
    xlen, xdim = x.size()
    if xlen >= padlen:
        return x
    if xdim == 0:
        return torch.empty([padlen, xdim], dtype=x.dtype)
    padding = padlen - xlen
    return F.pad(x, (0, 0, 0, padding), value=value)

def pad_attn_bias(x, padlen): # -> [padlen, padlen]
    xlen = x.size(0)
    if xlen >= padlen:
        return x
    padding = padlen - xlen
    x = F.pad(x, (0, padding, 0, padding), value=float("-inf"))
    x[xlen:, :xlen] = 0
    return x

def pad_spatial_pos(x, padlen): # -> [padlen, padlen]
    xlen = x.size(0)
    if xlen >= padlen:
        return x
    padding = padlen - xlen
    return F.pad(x, (0, padding, 0, padding), value=0)

def get_length_mask(lengths, max_len=None, dtype=torch.bool):
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths)
    if max_len is None:
        max_len = lengths.max().item()
    
    indices = torch.arange(max_len, device=lengths.device)
    mask = indices.unsqueeze(0) < lengths.unsqueeze(1)
    return mask.to(dtype)

def add_batch_indices(indices):
    num_indices = [index.size(1) for index in indices]
    batch_i = [torch.full([1, num], fill_value=i) for i, num in enumerate(num_indices)]
    batch_i = torch.cat(batch_i, dim=1)
    indices = torch.cat(indices, dim=1)
    return torch.cat([batch_i, indices])

def collate(graphs, spatial_pos_max=20, mask_unreachable=False):
    max_node_num = max(graph.num_nodes for graph in graphs)

    xs = torch.stack([pad_2d(graph.x, max_node_num) for graph in graphs])
    length_mask = get_length_mask([graph.num_nodes for graph in graphs], max_len=max_node_num, dtype=torch.float)
    edge_indices = add_batch_indices([graph.edge_index for graph in graphs])
    edge_attrs = torch.cat([graph.edge_attr for graph in graphs])
    path_indices = add_batch_indices([graph.path_indices for graph in graphs])

    attn_biases, spatial_poses = [], []
    for graph in graphs:
        N = graph.num_nodes
        spatial_pos = graph.spatial_pos
        
        attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token
        if mask_unreachable:
            attn_bias[1:, 1:][spatial_pos > spatial_pos_max] = float("-inf")
        attn_biases.append(pad_attn_bias(attn_bias, max_node_num + 1))
        
        # nodes are connected but too far
        spatial_pos[(spatial_pos > spatial_pos_max) & (spatial_pos < 510)] = spatial_pos_max + 1
        # nodes are unconnected
        spatial_pos[spatial_pos == 510] = spatial_pos_max + 2
        spatial_poses.append(pad_spatial_pos(spatial_pos + 1, max_node_num))

    batch = dict(x=xs,
                 length_mask=length_mask,
                 edge_index=edge_indices,
                 edge_attr=edge_attrs,
                 path_index=path_indices,
                 attn_bias=torch.stack(attn_biases),
                 spatial_pos=torch.stack(spatial_poses))
    return batch
