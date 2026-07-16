import os
import torch
import pandas as pd
from functools import partial
from torch.utils.data import Dataset
from .base import Batch
from .graphormer import RxnDataset
from ...utils import read_json, read_fasta

class EnzymeDataset(Dataset):
    def __init__(
        self,
        db_path, # path to a json/fasta file that maps enzyme ids to enzyme sequences
        enz_ids=None,
        tokenizer=None,
        max_length=None
    ):
        db_ext = os.path.splitext(db_path)[1]
        if db_ext in {'.fasta', '.faa', '.fa'}:
            self.enz_seqs = read_fasta(db_path)
        elif db_ext == '.json':
            self.enz_seqs = read_json(db_path)
        else:
            raise ValueError('Enzyme database must be a fasta or json file.')
        
        if enz_ids is None:
            self.enz_ids = pd.Index(list(self.enz_seqs.keys()))
        else:
            self.enz_ids = enz_ids
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.enz_ids)
    
    def get(self, enz_id):
        return self.enz_seqs[enz_id]
    
    def __getitem__(self, index):
        enz_id = self.enz_ids[index]
        return self.get(enz_id)
    
    @staticmethod
    def collate(raw_batch, tokenizer, max_length):
        batch = tokenizer(
            raw_batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return batch
    
    @property
    def collate_fn(self):
        return partial(
            self.collate,
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )

class RxnzymeDataset(Dataset):
    def __init__(
        self,
        rxn_db_dir,
        enz_db_path,
        rxn_enz_ids, # a pd.DataFrame with pairs of (rxn_id, enz_id)
        tokenizer,
        spatial_pos_max=5,
        mask_unreachable=False,
        max_length=None
    ):
        self.rxn_enz_ids = rxn_enz_ids
        self.rxn_dataset = RxnDataset(
            rxn_db_dir,
            rxn_ids=rxn_enz_ids['rxn_id'].unique(),
            spatial_pos_max=spatial_pos_max,
            mask_unreachable=mask_unreachable
        )
        self.enz_dataset = EnzymeDataset(
            enz_db_path,
            enz_ids=rxn_enz_ids['enz_id'].unique(),
            tokenizer=tokenizer,
            max_length=max_length
        )
        
    def init_lmdb(self):
        self.rxn_dataset.init_lmdb()
    
    def close_lmdb(self):
        self.rxn_dataset.close_lmdb()
    
    def __len__(self):
        return len(self.rxn_enz_ids)
    
    def __getitem__(self, index):
        row = self.rxn_enz_ids.iloc[index]
        rxn_graphs = self.rxn_dataset.get(row['rxn_id'])
        enz_seq = self.enz_dataset.get(row['enz_id'])
        return rxn_graphs, enz_seq
    
    @staticmethod
    def collate(raw_batch, rxn_collator, enz_collator):
        rxns, enzymes = zip(*raw_batch)
        batch = Batch(
            rxns=rxn_collator(rxns),
            enzymes=enz_collator(enzymes)
        )
        return batch

    @property
    def collate_fn(self):
        return partial(
            self.collate,
            rxn_collator=self.rxn_dataset.collate_fn,
            enz_collator=self.enz_dataset.collate_fn
        )

class RxnzymeDatasetForLTR(RxnzymeDataset):
    def __init__(
        self,
        rxn_db_dir,
        enz_db_path,
        labels, # a pd.DataFrame with triplets of (rxn_id, enz_id, label)
        tokenizer,
        spatial_pos_max=5,
        mask_unreachable=False,
        max_length=None
    ):
        super().__init__(
            rxn_db_dir,
            enz_db_path,
            labels[['rxn_id', 'enz_id']],
            tokenizer,
            spatial_pos_max,
            mask_unreachable,
            max_length
        )
        self.labels = labels
    
    def __getitem__(self, index):
        row = self.labels.iloc[index]
        rxn_graphs, enz_seq = super().__getitem__(index)
        return rxn_graphs, enz_seq, row['label']
    
    @staticmethod
    def collate(raw_batch, rxn_collator, enz_collator):
        rxns, enzymes, labels = zip(*raw_batch)
        batch = Batch(
            rxns=rxn_collator(rxns),
            enzymes=enz_collator(enzymes),
            labels=torch.tensor(labels, dtype=torch.float)
        )
        return batch
