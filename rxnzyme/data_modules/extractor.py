import torch
from lightning import LightningDataModule
from .base import get_regular_sampler, get_dataloader
from .prorxn import get_eval_labels
from ..data.datasets import ignore_label, RxnDataset, EnzymeDataset, RxnzymeDataset

class ExtractorDataModule(LightningDataModule):
    def __init__(
            self,
            rxn_db_dir=None,
            enz_db_path=None,
            tokenizer=None,
            train_config=None,
            pair_ids=None
    ):
        super().__init__()
        self.rxn_db_dir = rxn_db_dir
        self.enz_db_path = enz_db_path
        self.tokenizer = tokenizer
        self.train_config = train_config
        self.pair_ids = pair_ids
    
    def setup(self, stage='predict'):
        if stage == 'predict':
            if self.pair_ids is not None:
                self.dataset = RxnzymeDataset(
                    rxn_db_dir=self.rxn_db_dir,
                    enz_db_path=self.enz_db_path,
                    rxn_enz_ids=self.pair_ids,
                    tokenizer=self.tokenizer,
                    spatial_pos_max=self.train_config['multi_hop_max_dist'],
                    mask_unreachable=False,
                    max_length=self.train_config['max_length']
                )
            
            elif self.rxn_db_dir is not None:
                self.dataset = RxnDataset(
                    db_dir=self.rxn_db_dir,
                    rxn_ids=None,
                    spatial_pos_max=self.train_config['multi_hop_max_dist'],
                    mask_unreachable=False
                )
            
            elif self.enz_db_path is not None:
                self.dataset = EnzymeDataset(
                    db_path=self.enz_db_path,
                    enz_ids=None,
                    tokenizer=self.tokenizer,
                    max_length=self.train_config['max_length']
                )
    
    def predict_dataloader(self):
        batch_sampler = get_regular_sampler(
            self.dataset,
            batch_size=self.train_config['eval_batch_size'],
            shuffle=False,
            rank=self.trainer.global_rank,
            world_size=self.trainer.world_size
        )
        return get_dataloader(
            self.dataset,
            batch_sampler=batch_sampler,
            num_workers=self.train_config['num_workers'],
            prefetch_factor=self.train_config['prefetch_factor']
        )

class DenseRetrieverDataModule(LightningDataModule):
    def __init__(
            self,
            enz_db_path,
            tokenizer,
            train_config,
            train_ids=None,
            test_ids=None,
            test_query_ids=None, # pair ids
            pred_query_ids=None # list of enz_ids
        ):
        super().__init__()
        self.enz_db_path = enz_db_path
        self.tokenizer = tokenizer
        self.train_config = train_config
        self.train_ids = train_ids
        self.test_ids = test_ids
        self.test_query_ids = test_query_ids
        self.pred_query_ids = pred_query_ids
    
    def setup(self, stage='test'):
        if stage == 'test':
            self.test_dataset = EnzymeDataset(
                db_path=self.enz_db_path,
                enz_ids=None,
                tokenizer=self.tokenizer,
                max_length=self.train_config['max_length']
            )
            
            self.test_labels, rxn_id_to_idx, enz_id_to_idx = get_eval_labels(
                eval_rxn_enz_ids=self.test_ids,
                eval_rxn_ids=self.test_ids['rxn_id'].unique(),
                eval_enz_ids=self.test_dataset.enz_ids,
                train_rxn_enz_ids=self.train_ids
            )

            test_query_ids = self.test_query_ids.copy()
            test_query_ids['rxn_idx'] = test_query_ids['rxn_id'].map(rxn_id_to_idx)
            test_query_ids['enz_idx'] = test_query_ids['enz_id'].map(enz_id_to_idx)
            test_query_ids.sort_values('rxn_idx', inplace=True)
            test_rxn_indices = torch.from_numpy(test_query_ids['rxn_idx'].values)
            test_query_indices = torch.from_numpy(test_query_ids['enz_idx'].values)
            
            self.test_labels[test_rxn_indices, test_query_indices] = ignore_label
            self.test_query_indices = test_query_indices
        
        elif stage == 'predict':
            self.pred_dataset = EnzymeDataset(
                db_path=self.enz_db_path,
                enz_ids=None,
                tokenizer=self.tokenizer,
                max_length=self.train_config['max_length']
            )
            enz_id_to_idx = {enz_id: idx for idx, enz_id in enumerate(self.pred_dataset.enz_ids)}
            self.pred_query_indices = torch.tensor([enz_id_to_idx[enz_id] for enz_id in self.pred_query_ids])
    
    def test_dataloader(self):
        batch_sampler = get_regular_sampler(
            self.test_dataset,
            batch_size=self.train_config['eval_batch_size'],
            shuffle=False,
            rank=self.trainer.global_rank,
            world_size=self.trainer.world_size
        )
        return get_dataloader(
            self.test_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.train_config['num_workers'],
            prefetch_factor=self.train_config['prefetch_factor']
        )
    
    def predict_dataloader(self):
        batch_sampler = get_regular_sampler(
            self.pred_dataset,
            batch_size=self.train_config['eval_batch_size'],
            shuffle=False,
            rank=self.trainer.global_rank,
            world_size=self.trainer.world_size
        )
        return get_dataloader(
            self.pred_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.train_config['num_workers'],
            prefetch_factor=self.train_config['prefetch_factor']
        )
