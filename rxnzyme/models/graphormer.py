import torch
from torch import nn
from .modules import get_activation_fn

class LMHead(nn.Module):
    def __init__(self, hidden_size, num_classes, activation_fn='gelu'):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation_fn = get_activation_fn(activation_fn)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.decoder = nn.Linear(hidden_size, num_classes)
    
    def forward(self, hiddens):
        x = self.dense(hiddens)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x

class ClsHead(nn.Module):
    def __init__(self, hidden_size, num_classes, activation_fn='gelu', dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation_fn = get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, hiddens):
        x = self.dropout(hiddens)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

class MolGraphormerForMLM(nn.Module):
    def __init__(self, graphormer, class_nums):
        super().__init__()
        self.graphormer = graphormer

        self.mlm_class_nums = list(class_nums)
        self.rcp_class_nums = self.mlm_class_nums + [2]
        total_mlm_classes = sum(self.mlm_class_nums)
        total_rcp_classes = sum(self.rcp_class_nums)

        # predict all labels simultaneously
        self.r_mlm_head = LMHead(
            hidden_size=graphormer.embedding_dim,
            activation_fn=graphormer.activation_fn,
            num_classes=total_mlm_classes
        )
        self.r_rcp_head = LMHead(
            hidden_size=graphormer.embedding_dim,
            activation_fn=graphormer.activation_fn,
            num_classes=total_rcp_classes
        )
        
        self.p_mlm_head = LMHead(
            hidden_size=graphormer.embedding_dim,
            activation_fn=graphormer.activation_fn,
            num_classes=total_mlm_classes
        )
        self.p_rcp_head = LMHead(
            hidden_size=graphormer.embedding_dim,
            activation_fn=graphormer.activation_fn,
            num_classes=total_rcp_classes
        )
        
        proj_dim = graphormer.embedding_dim // 2
        proj_hidden_dim = (graphormer.embedding_dim + proj_dim) // 2
        self.proj = nn.Sequential(
                nn.Linear(graphormer.embedding_dim, proj_hidden_dim, bias=False),
                get_activation_fn(graphormer.activation_fn),
                nn.Linear(proj_hidden_dim, proj_dim, bias=False)
            )
        
    def forward(self, batch):
        r_hiddens, r_reps = self.graphormer(batch['reactants'])
        r_reps = self.proj(r_reps)
        r_mlm_logits = self.r_mlm_head(r_hiddens)
        r_rcp_logits = self.r_rcp_head(r_hiddens)

        p_hiddens, p_reps = self.graphormer(batch['products'])
        p_reps = self.proj(p_reps)
        p_mlm_logits = self.p_mlm_head(p_hiddens)
        p_rcp_logits = self.p_rcp_head(p_hiddens)

        return r_mlm_logits, r_rcp_logits, r_reps, p_mlm_logits, p_rcp_logits, p_reps
    
class RxnGraphormer(nn.Module):
    def __init__(self, mol_graphormer, cgr_graphormer=None):
        super().__init__()
        self.mol_graphormer = mol_graphormer
        self.cgr_graphormer = cgr_graphormer
        self.activation_fn = mol_graphormer.activation_fn
        self.dropout = mol_graphormer.dropout

        if cgr_graphormer is None:
            self.rep_out = nn.Linear(
                mol_graphormer.embedding_dim * 2,
                mol_graphormer.embedding_dim,
                bias=False
            )
            self.embedding_dim = mol_graphormer.embedding_dim
        else:
            self.embedding_dim = cgr_graphormer.embedding_dim

    def forward(self, batch):
        r_hiddens, r_reps = self.mol_graphormer(batch['reactants'])
        p_hiddens, p_reps = self.mol_graphormer(batch['products'])

        if self.cgr_graphormer is None:
            rxn_hiddens = torch.cat([r_hiddens, p_hiddens], dim=1)
            rxn_reps = torch.cat([r_reps, p_reps], dim=1)
            rxn_reps = self.rep_out(rxn_reps)
            batch['length_mask'] = torch.cat(
                [batch['reactants']['length_mask'], batch['products']['length_mask']], dim=1
            )
        else:
            max_node_num = batch['cgrs']['x'].size(1)
            batch['cgrs']['x'] = torch.cat(
                [r_hiddens[:, :max_node_num], p_hiddens[:, :max_node_num]], dim=-1
            )
            rxn_hiddens, rxn_reps = self.cgr_graphormer(batch['cgrs'])
            batch['length_mask'] = batch['cgrs']['length_mask']
        
        return rxn_hiddens, rxn_reps

class RxnGraphormerForSCL(nn.Module):
    def __init__(self, rxn_graphormer, proj_dim, proj_type='linear'):
        super().__init__()
        self.rxn_graphormer = rxn_graphormer

        if proj_type == 'linear':
            self.proj = nn.Linear(rxn_graphormer.embedding_dim, proj_dim, bias=False)
        elif proj_type == 'nonlinear':
            proj_hidden_dim = (rxn_graphormer.embedding_dim + proj_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(rxn_graphormer.embedding_dim, proj_hidden_dim, bias=False),
                get_activation_fn(rxn_graphormer.activation_fn),
                nn.Linear(proj_hidden_dim, proj_dim, bias=False)
            )
        else:
            raise ValueError(f'Unknown projection type: {proj_type}')

    def forward(self, batch):
        _, rxn_reps = self.rxn_graphormer(batch)
        rxn_reps = self.proj(rxn_reps)
        return rxn_reps

class RxnGraphormerForCls(nn.Module):
    def __init__(self, rxn_graphormer, num_classes):
        super().__init__()
        self.rxn_graphormer = rxn_graphormer
        self.cls_head = ClsHead(
            hidden_size=rxn_graphormer.embedding_dim,
            num_classes=num_classes,
            activation_fn=rxn_graphormer.activation_fn,
            dropout=rxn_graphormer.dropout
        )

    def forward(self, batch):
        _, rxn_reps = self.rxn_graphormer(batch)
        logits = self.cls_head(rxn_reps)
        return logits
