from typing import Optional, Tuple

import torch
import torch.nn as nn

from .multihead_attention import MultiheadAttention
from .graphormer_layers import GraphNodeFeature, GraphAttnBias
from .graphormer_graph_encoder_layer import GraphormerGraphEncoderLayer


class CustomDropout(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(x)


class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        return self.layer_norm(x)


class CustomLayerDropModuleList(nn.ModuleList):
    def __init__(self, modules=None, p=0.0):
        super().__init__(modules)
        self.layerdrop = p

    def forward(self, x, *args, **kwargs):
        for module in self:
            if self.layerdrop == 0 or torch.rand(1).item() >= self.layerdrop:
                x = module(x, *args, **kwargs)
        return x


def init_graphormer_params(module):
    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class GraphormerGraphEncoder(nn.Module):
    def __init__(
        self,
        num_spatial: int,
        num_edge_dis: int,
        edge_type: str,
        feature_dim: int,
        edge_attr_dim: int,
        num_encoder_layers: int = 12,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 768,
        num_attention_heads: int = 32,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        encoder_normalize_before: bool = False,
        pre_layernorm: bool = False,
        apply_graphormer_init: bool = False,
        activation_fn: str = "gelu",
        pooling: str = "cls",
        embed_scale: float = None,
    ) -> None:

        super().__init__()
        self.dropout = dropout
        self.dropout_module = CustomDropout(self.dropout)
        self.layerdrop = layerdrop
        self.embedding_dim = embedding_dim
        self.apply_graphormer_init = apply_graphormer_init
        self.activation_fn = activation_fn

        if pooling not in {"cls", "mean", "max"}:
            raise ValueError(f"Unknown pooling type: {pooling}. Must be one of {{cls, mean, max}}.")
        self.pooling = pooling

        self.graph_node_feature = GraphNodeFeature(
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
            feature_dim=feature_dim,
        )

        self.graph_attn_bias = GraphAttnBias(
            num_heads=num_attention_heads,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            edge_attr_dim=edge_attr_dim,
            n_layers=num_encoder_layers,
        )

        self.embed_scale = embed_scale
        self.emb_layer_norm = CustomLayerNorm(self.embedding_dim) if encoder_normalize_before else None
        self.final_layer_norm = CustomLayerNorm(self.embedding_dim) if pre_layernorm else None

        if self.layerdrop > 0.0:
            self.layers = CustomLayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                GraphormerGraphEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.dropout.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=self.activation_fn,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        if self.apply_graphormer_init:
            self.apply(init_graphormer_params)

    def readout(self, x, length_mask):
        if self.pooling == "cls":
            return x[:, 0, :]
        
        x = x[:, 1:, :]
        if self.pooling == "mean":
            x = x * length_mask.unsqueeze(-1)
            graph_rep = x.sum(dim=1) / length_mask.sum(dim=1, keepdim=True)
        else:
            x.masked_fill_(length_mask.unsqueeze(-1) == 0, float('-inf'))
            graph_rep = x.max(dim=1).values
        return graph_rep

    def forward(
        self,
        batched_data,
        perturb=None,
        last_state_only: bool = True,
        token_embeddings: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        data_x = batched_data["x"]
        n_graph = data_x.size(0)
        padding_mask = batched_data["length_mask"] == 0
        padding_mask_cls = torch.zeros(
            n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype
        )
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)

        x = token_embeddings if token_embeddings is not None else self.graph_node_feature(batched_data)
        if perturb is not None:
            x[:, 1:, :] += perturb

        attn_bias = self.graph_attn_bias(batched_data)

        if self.embed_scale is not None:
            x = x * self.embed_scale

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)
        x = x.transpose(0, 1)

        inner_states = []
        for layer in self.layers:
            x, _ = layer(
                x,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                self_attn_bias=attn_bias,
            )
            if not last_state_only:
                inner_states.append(x.transpose(0, 1))

        x = x.transpose(0, 1)
        graph_rep = self.readout(x, batched_data["length_mask"])
        x = x[:, 1:, :]

        if last_state_only:
            return x, graph_rep
        else:
            return x, graph_rep, inner_states
