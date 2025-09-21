import math

import torch
import torch.nn as nn


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class GraphNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph.
    """

    def __init__(
        self, hidden_dim, n_layers, feature_dim
    ):
        super(GraphNodeFeature, self).__init__()
        # 1 for graph token; degrees are combined with the input fearues
        self.feature_encoder = nn.Linear(feature_dim, hidden_dim, bias=False)  # make padded feature still 0

        self.graph_token = nn.Embedding(1, hidden_dim)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        x = batched_data["x"]
        n_graph = x.size(0)

        # node feauture + graph token
        node_feature = self.feature_encoder(x)

        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)

        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)

        return graph_node_feature    


class GraphAttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """

    def __init__(
        self,
        num_heads,
        num_spatial,
        num_edge_dis,
        edge_type,
        edge_attr_dim,
        n_layers,
    ):
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads

        self.edge_encoder = nn.Linear(edge_attr_dim, num_heads, bias=False)
        self.edge_type = edge_type
        if self.edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(
                num_edge_dis * num_heads * num_heads, 1
            )
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)

        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data):
        x, edge_index, edge_attr, path_index, attn_bias, spatial_pos = (
            batched_data["x"],
            batched_data["edge_index"],
            batched_data["edge_attr"],
            batched_data["path_index"],
            batched_data["attn_bias"],
            batched_data["spatial_pos"],
        )        
        n_graph, n_node = x.shape[:2]
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1
        ) # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # reset spatial pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t
        
        # -> [n_total_edges, n_heads]
        edge_attr = self.edge_encoder(edge_attr)
        # dense_adj shape: [n_graph, n_node, n_node, n_head]
        dense_adj = torch.zeros(
            *spatial_pos.size(), edge_attr.size(1),
            dtype=edge_attr.dtype, device=edge_attr.device
        )
        dense_adj[edge_index[0], edge_index[1], edge_index[2]] = edge_attr

        # edge feature
        if self.edge_type == "multi_hop":
            # edge_input shape: [n_graph, n_node, n_node, max_dist, n_head]
            max_dist = path_index[3].max() + 1
            edge_input = torch.zeros(
                *dense_adj.shape[:3], max_dist, dense_adj.shape[3],
                dtype=dense_adj.dtype, device=dense_adj.device
            )
            edge_input[path_index[0], path_index[1], path_index[2], path_index[3]] = \
                dense_adj[path_index[0], path_index[4], path_index[5]]

            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, x > 1 to x - 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            spatial_pos_ = spatial_pos_.clamp(0, max_dist)

            # -> [max_dist, (n_graph * n_node * n_node), n_head]
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(
                max_dist, -1, self.num_heads
            )
            edge_input_flat = torch.bmm(
                edge_input_flat,
                self.edge_dis_encoder.weight.reshape(
                    -1, self.num_heads, self.num_heads
                )[:max_dist, :, :],
            )
            # -> [max_dist, n_graph, n_node, n_node, n_head]
            edge_input = edge_input_flat.reshape(
                max_dist, n_graph, n_node, n_node, self.num_heads
            ).permute(1, 2, 3, 0, 4)

            edge_input = (
                edge_input.sum(-2) / (spatial_pos_.float().unsqueeze(-1))
            ).permute(0, 3, 1, 2)
        else:
            # -> [n_graph, n_head, n_node, n_node]
            edge_input = dense_adj.permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        return graph_attn_bias
