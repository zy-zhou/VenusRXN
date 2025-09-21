import os
import atexit
import torch
import lmdb
import pickle
import random
import pandas as pd
from functools import partial
from torch.utils.data import Dataset
from .reaction import x_vocabs, one_hots_to_labels
from .graph_utils.collator import pad_2d, collate
from .database import decompress_mol_graph, get_db_size
from ..utils import read_json

ignore_label = -1

class Batch(dict):
    @staticmethod
    def pin_batch(batch):
        for key, value in batch.items():
            if type(value) is torch.Tensor:
                batch[key] = value.pin_memory()
            elif isinstance(value, dict):
                Batch.pin_batch(value)

    def pin_memory(self):
        Batch.pin_batch(self)

class LMDBDataset(Dataset):
    def __init__(self, db_dir, keys):
        self.db_dir = db_dir
        self.keys = keys
        self.env = None
        self.txn = None
        atexit.register(self.close_lmdb)
    
    def init_lmdb(self):
        if self.env is None:
            map_size = get_db_size(self.db_dir)
            self.env = lmdb.open(
                self.db_dir,
                map_size=map_size,
                readonly=True,
                lock=False
            )
            self.txn = self.env.begin()
    
    def close_lmdb(self):
        if self.env is not None:
            if self.txn is not None:
                self.txn.commit()
            self.env.close()
            self.env = None
            self.txn = None
    
    def __len__(self):
        return len(self.keys)
    
    def get(self, key):
        value = self.txn.get(str(key).encode())
        value = pickle.loads(value)
        return value

    def __getitem__(self, index):
        key = self.keys[index]
        return self.get(key)

class RxnDataset(LMDBDataset):
    def __init__(
            self,
            db_dir,
            rxn_ids=None,
            spatial_pos_max=20,
            mask_unreachable=False
        ):
        if rxn_ids is None:
            rxn_ids = pd.read_csv(
                os.path.join(db_dir, 'metadata.csv'),
                index_col='rxn_id',
                usecols=['rxn_id']
            ).index
        super().__init__(db_dir, rxn_ids)
        self.spatial_pos_max = spatial_pos_max
        self.mask_unreachable = mask_unreachable

    def get(self, rxn_id):
        rxn_graphs = super().get(rxn_id)
        if type(rxn_graphs) is dict:
            reactants_graph = decompress_mol_graph(rxn_graphs['reactants'])
            products_graph = decompress_mol_graph(rxn_graphs['products'])
            cgr = decompress_mol_graph(rxn_graphs['cgr'])
            return reactants_graph, products_graph, cgr
        else: # the batch contains reactants or products only
            return decompress_mol_graph(rxn_graphs)

    @staticmethod
    def collate(raw_batch, spatial_pos_max, mask_unreachable):
        if type(raw_batch[0]) is tuple:
            reactants, products, cgrs = zip(*raw_batch)
            batch = Batch(
                reactants=collate(reactants, spatial_pos_max, mask_unreachable),
                products=collate(products, spatial_pos_max, mask_unreachable),
                cgrs=collate(cgrs, spatial_pos_max, mask_unreachable)
            )
        else: # the batch contains reactants or products only
            batch = Batch(collate(raw_batch, spatial_pos_max, mask_unreachable))
        return batch
   
    @property
    def collate_fn(self):
        return partial(
            self.collate,
            spatial_pos_max=self.spatial_pos_max,
            mask_unreachable=self.mask_unreachable
        )

class RxnDatasetForMLM(RxnDataset):
    def __init__(
            self,
            db_dir,
            rxn_ids=None,
            mask_rate=0.15,
            spatial_pos_max=20,
            mask_unreachable=False
        ):
        super().__init__(db_dir, rxn_ids, spatial_pos_max, mask_unreachable)
        self.mask_rate = mask_rate
        self.vocab_sizes = list(map(len, x_vocabs.values()))
        self.vocab_offsets = torch.cumsum(torch.tensor([0] + self.vocab_sizes[:-1]), dim=0)

        label_counts = torch.load(os.path.join(db_dir, 'label_counts.pkl'))
        self.attr_combs = []
        self.label_maps = []
        for label_count in label_counts:
            self.attr_combs.append(label_count['attr_comb'])
            # labels shape: [num_unique, comb_size]
            labels = torch.tensor(list(label_count['count'].keys()))
            size = labels.max(0).values + 1
            label_map = torch.zeros(size.tolist(), dtype=torch.long)
            label_map[list(labels.T)] = torch.arange(labels.size(0))
            self.label_maps.append(label_map)

    def get_mlm_labels(self, x):
        x = one_hots_to_labels(x, x_vocabs)
        mlm_labels = [label_map[list(x[:, attr_comb].T)] \
                          for attr_comb, label_map in zip(self.attr_combs, self.label_maps)]
        return torch.stack(mlm_labels, dim=1)

    def mask_nodes(self, x):
        # in-place mask
        ignore_indices = []
        for i in range(x.size(0)):
            p = random.random()

            if p < self.mask_rate:
                p /= self.mask_rate
                x[i] = 0
            
                if p < 0.8: # 80% randomly change token to mask token
                    x[i, -1] = 1
                elif p < 0.9: # 10% randomly change token to random token
                    randoms = torch.tensor([random.randrange(size) for size in self.vocab_sizes])
                    x[i, self.vocab_offsets + randoms] = 1
            
            else:
                ignore_indices.append(i)
        
        return torch.tensor(ignore_indices, dtype=torch.long)
    
    def get(self, rxn_id):
        rxn_graphs = super(RxnDataset, self).get(rxn_id)

        reactants_graph = decompress_mol_graph(rxn_graphs['reactants'])
        products_graph = decompress_mol_graph(rxn_graphs['products'])
        N_AAM = reactants_graph.num_aam_nodes
        
        r_mlm_labels = self.get_mlm_labels(reactants_graph.x)
        p_rcp_labels = r_mlm_labels[:N_AAM].clone()
        ignore_indices = self.mask_nodes(reactants_graph.x)
        r_mlm_labels[ignore_indices] = ignore_label
        
        p_mlm_labels = self.get_mlm_labels(products_graph.x)
        r_rcp_labels = p_mlm_labels[:N_AAM].clone()
        ignore_indices = self.mask_nodes(products_graph.x)
        p_mlm_labels[ignore_indices] = ignore_label

        reactive_center = rxn_graphs['reactive_center'].long()
        center_labels = torch.zeros((N_AAM, 1), dtype=torch.long)
        center_labels[reactive_center] = 1
        ignore_indices = ~ center_labels.squeeze(1).bool()
        r_rcp_labels[ignore_indices] = ignore_label
        p_rcp_labels[ignore_indices] = ignore_label
        r_rcp_labels = torch.cat([r_rcp_labels, center_labels], dim=1)
        p_rcp_labels = torch.cat([p_rcp_labels, center_labels], dim=1)

        return (
            reactants_graph,
            products_graph,
            r_mlm_labels,
            r_rcp_labels,
            p_mlm_labels,
            p_rcp_labels
        )

    @staticmethod
    def collate(raw_batch, spatial_pos_max, mask_unreachable):
        (
            reactants,
            products,
            r_mlm_labels,
            r_rcp_labels,
            p_mlm_labels,
            p_rcp_labels
        ) = zip(*raw_batch)
        
        batch = Batch(
            reactants=collate(reactants, spatial_pos_max, mask_unreachable),
            products=collate(products, spatial_pos_max, mask_unreachable)
        )
        
        padlen = batch['reactants']['x'].size(1)
        batch['r_mlm_labels'] = torch.stack(
            [pad_2d(labels, padlen, value=ignore_label) for labels in r_mlm_labels]
        )
        batch['r_rcp_labels'] = torch.stack(
            [pad_2d(labels, padlen, value=ignore_label) for labels in r_rcp_labels]
        )
        
        padlen = batch['products']['x'].size(1)
        batch['p_mlm_labels'] = torch.stack(
            [pad_2d(labels, padlen, value=ignore_label) for labels in p_mlm_labels]
        )
        batch['p_rcp_labels'] = torch.stack(
            [pad_2d(labels, padlen, value=ignore_label) for labels in p_rcp_labels]
        )
        return batch

class RxnDatasetForCls(RxnDataset):
    def __init__(
            self,
            db_dir,
            labels, # a pd.Series that maps reaction ids to labels
            spatial_pos_max=5,
            mask_unreachable=False
        ):
        super().__init__(db_dir, labels.index)
        self.labels = labels
        self.spatial_pos_max = spatial_pos_max
        self.mask_unreachable = mask_unreachable

    def get(self, rxn_id):
        reactants_graph, products_graph, cgr_graph = super().get(rxn_id)
        label = self.labels[rxn_id]
        return reactants_graph, products_graph, cgr_graph, label

    @staticmethod
    def collate(raw_batch, spatial_pos_max, mask_unreachable):
        reactants, products, cgrs, labels = zip(*raw_batch)
        batch = Batch(
            reactants=collate(reactants, spatial_pos_max, mask_unreachable),
            products=collate(products, spatial_pos_max, mask_unreachable),
            cgrs=collate(cgrs, spatial_pos_max, mask_unreachable),
            labels=torch.tensor(labels, dtype=torch.long)
        )
        return batch

class EnzymeDataset(Dataset):
    def __init__(
            self,
            db_path, # path to a json file containing a dict that maps enzyme ids to enzyme sequences
            enz_ids=None,
            tokenizer=None,
            max_length=None
    ):
        self.enz_seqs = read_json(db_path)
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
        label = row['label']
        return rxn_graphs, enz_seq, label
    
    @staticmethod
    def collate(raw_batch, rxn_collator, enz_collator):
        rxns, enzymes, labels = zip(*raw_batch)
        batch = Batch(
            rxns=rxn_collator(rxns),
            enzymes=enz_collator(enzymes),
            labels=torch.tensor(labels, dtype=torch.float)
        )
        return batch
