import torch
import torch.nn as nn
from .modules import get_activation_fn

class RankingHead(nn.Module):
    def __init__(self, hidden_size, activation_fn='tanh', dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation_fn = get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(dropout)
        self.ranker = nn.Linear(hidden_size, 1)

    def forward(self, hiddens):
        x = self.dropout(hiddens)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.ranker(x)
        return x

class ProRxn(nn.Module):
    def __init__(self, rxn_graphormer, plm, proj_dim, proj_type='linear'):
        super().__init__()
        self.rxn_graphormer = rxn_graphormer
        self.plm = plm
        rxn_embed_dim = rxn_graphormer.embedding_dim
        enz_embed_dim = plm.config.hidden_size

        if proj_type == 'linear':
            self.rxn_proj = nn.Linear(rxn_embed_dim, proj_dim, bias=False)
            self.enz_proj = nn.Linear(enz_embed_dim, proj_dim, bias=False)
        elif proj_type == 'nonlinear':
            proj_hidden_dim = (rxn_embed_dim + proj_dim) // 2
            self.rxn_proj = nn.Sequential(
                nn.Linear(rxn_embed_dim, proj_hidden_dim, bias=False),
                get_activation_fn(rxn_graphormer.activation_fn),
                nn.Linear(proj_hidden_dim, proj_dim, bias=False)
            )

            proj_hidden_dim = (enz_embed_dim + proj_dim) // 2
            self.enz_proj = nn.Sequential(
                nn.Linear(enz_embed_dim, proj_hidden_dim, bias=False),
                get_activation_fn(rxn_graphormer.activation_fn),
                nn.Linear(proj_hidden_dim, proj_dim, bias=False)
            )
        else:
            raise ValueError(f'Unknown projection type: {proj_type}')
        
        if self.plm.config.add_cross_attention:
            self.out_proj = nn.Linear(plm.config.hidden_size, 1)
    
    def encode_rxns(self, rxn_batch, use_proj=True, return_full_hiddens=False):
        rxn_hiddens, rxn_reps = self.rxn_graphormer(rxn_batch)
        if use_proj:
            rxn_reps = self.rxn_proj(rxn_reps)
        return (rxn_hiddens, rxn_reps) if return_full_hiddens else rxn_reps

    def encode_enzymes(self, enz_batch, use_proj=True, return_full_hiddens=False):
        enz_hiddens = self.plm(**enz_batch).last_hidden_state
        enz_reps = enz_hiddens[:, 0] # cls token
        if use_proj:
            enz_reps = self.enz_proj(enz_reps)
        return (enz_hiddens, enz_reps) if return_full_hiddens else enz_reps
    
    def encode_rxn_enzs(self, rxn_enz_batch, use_proj=True, return_full_hiddens=False):
        if not self.plm.config.add_cross_attention:
            raise ValueError('The PLM should be configured with cross attention layers.')
        
        rxn_hiddens, _ = self.rxn_graphormer(rxn_enz_batch['rxns'])
        outputs = self.plm(
            **rxn_enz_batch['enzymes'],
            encoder_hidden_states=rxn_hiddens,
            encoder_attention_mask=rxn_enz_batch['rxns']['length_mask']
        )
        rxn_enz_hiddens = outputs.last_hidden_state
        
        if use_proj:
            rxn_enz_reps = outputs.pooler_output
        else:
            rxn_enz_reps = rxn_enz_hiddens[:, 0]
        return (rxn_enz_hiddens, rxn_enz_reps) if return_full_hiddens else rxn_enz_reps
    
    def forward(self, batch, ranking=False):
        if ranking:
            rxn_enz_reps = self.encode_rxn_enzs(batch)
            logits = self.out_proj(rxn_enz_reps).squeeze(1)
            return logits
        else:
            rxn_hiddens, rxn_reps = self.encode_rxns(batch['rxns'], return_full_hiddens=True)
            enz_hiddens, enz_reps = self.encode_enzymes(batch['enzymes'], return_full_hiddens=True)
            return rxn_hiddens, rxn_reps, enz_hiddens, enz_reps

class ProRxnForLTR(nn.Module):
    def __init__(self, prorxn, output_type='cross_attn'):
        super().__init__()
        self.prorxn = prorxn
        self.output_type = output_type
        rxn_embed_dim = prorxn.rxn_graphormer.embedding_dim
        enz_embed_dim = prorxn.plm.config.hidden_size
        activation_fn = prorxn.rxn_graphormer.activation_fn
        dropout = prorxn.rxn_graphormer.dropout
        
        if output_type == 'rxn':
            self.ranking_head = RankingHead(
                hidden_size=rxn_embed_dim,
                activation_fn=activation_fn,
                dropout=dropout
            )
        elif output_type == 'enzyme':
            self.ranking_head = RankingHead(
                hidden_size=enz_embed_dim,
                activation_fn='tanh',
                dropout=dropout
            )
        elif output_type == 'concat':
            self.ranking_head = RankingHead(
                hidden_size=rxn_embed_dim + enz_embed_dim,
                activation_fn='tanh',
                dropout=dropout
            )
        elif output_type not in {'cosine', 'cross_attn'}:
            raise ValueError(
                f'Unknown output type: {output_type}. Must be one of {{rxn, enzyme, concat, cosine, cross_attn}}.'
            )
        
    def forward(self, batch):
        if self.output_type == 'rxn':
            rxn_reps = self.prorxn.encode_rxns(batch['rxns'], use_proj=False)
            outputs = self.ranking_head(rxn_reps).squeeze(1)
        
        elif self.output_type == 'enzyme':
            enz_reps = self.prorxn.encode_enzymes(batch['enzymes'], use_proj=False)
            outputs = self.ranking_head(enz_reps).squeeze(1)
        
        elif self.output_type == 'concat':
            rxn_reps = self.prorxn.encode_rxns(batch['rxns'], use_proj=False)
            enz_reps = self.prorxn.encode_enzymes(batch['enzymes'], use_proj=False)
            rxn_enz_reps = torch.cat([rxn_reps, enz_reps], dim=1)
            outputs = self.ranking_head(rxn_enz_reps).squeeze(1)
        
        elif self.output_type == 'cosine':
            rxn_reps = self.prorxn.encode_rxns(batch['rxns'])
            enz_reps = self.prorxn.encode_enzymes(batch['enzymes'])
            outputs = torch.cosine_similarity(rxn_reps, enz_reps)
        
        else:
            outputs = self.prorxn(batch, ranking=True)
        
        return outputs
