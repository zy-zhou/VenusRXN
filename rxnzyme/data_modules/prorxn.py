import torch
import pandas as pd
from lightning import LightningDataModule
from .base import get_regular_sampler, get_dataloader
from ..data.datasets import (
    ignore_label,
    RxnDataset,
    EnzymeDataset,
    RxnzymeDataset,
    RxnzymeDatasetForLTR
)
from ..data.samplers import (
    RxnzymeSamplerForCL,
    DistributedRxnzymeSamplerForCL
)

def get_eval_labels(eval_rxn_enz_ids, eval_rxn_ids, eval_enz_ids, train_rxn_enz_ids=None):
    rxn_id_to_idx = {rxn_id: idx for idx, rxn_id in enumerate(eval_rxn_ids)}
    rxn_indices = torch.from_numpy(eval_rxn_enz_ids['rxn_id'].map(rxn_id_to_idx).values)
    enz_id_to_idx = {enz_id: idx for idx, enz_id in enumerate(eval_enz_ids)}
    enz_indices = torch.from_numpy(eval_rxn_enz_ids['enz_id'].map(enz_id_to_idx).values)
    
    labels = torch.zeros((len(rxn_id_to_idx), len(enz_id_to_idx)), dtype=torch.float)
    labels[rxn_indices, enz_indices] = 1

    if train_rxn_enz_ids is not None:
        train_rxn_enz_ids = train_rxn_enz_ids[train_rxn_enz_ids['rxn_id'].isin(rxn_id_to_idx.keys())]
        rxn_indices = torch.from_numpy(train_rxn_enz_ids['rxn_id'].map(rxn_id_to_idx).values)
        enz_indices = torch.from_numpy(train_rxn_enz_ids['enz_id'].map(enz_id_to_idx).values)
        labels[rxn_indices, enz_indices] = ignore_label

    return labels, rxn_id_to_idx, enz_id_to_idx
    
class ProRxnDataModule(LightningDataModule):
    def __init__(
            self,
            rxn_db_dir,
            enz_db_path,
            tokenizer,
            train_config,
            train_ids=None,
            val_ids=None,
            test_ids=None,
            test_cdts=None
        ):
        super().__init__()
        self.rxn_db_dir = rxn_db_dir
        self.enz_db_path = enz_db_path
        self.train_config = train_config
        self.tokenizer = tokenizer
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids
        self.test_cdts = test_cdts

    def setup(self, stage='fit'):
        if stage == 'fit':
            self.train_dataset = RxnzymeDataset(
                rxn_db_dir=self.rxn_db_dir,
                enz_db_path=self.enz_db_path,
                rxn_enz_ids=self.train_ids,
                tokenizer=self.tokenizer,
                spatial_pos_max=self.train_config['multi_hop_max_dist'],
                mask_unreachable=False,
                max_length=self.train_config['max_length']
            )

        if stage in {'fit', 'validate'}:
            if self.val_ids is None or len(self.val_ids) == 0:
                self.val_rxn_dataset = self.val_enz_dataset = None
            else:
                self.val_rxn_dataset = RxnDataset(
                    db_dir=self.rxn_db_dir,
                    rxn_ids=self.val_ids['rxn_id'].unique(),
                    spatial_pos_max=self.train_config['multi_hop_max_dist'],
                    mask_unreachable=False
                )

                self.val_enz_dataset = EnzymeDataset(
                    db_path=self.enz_db_path,
                    enz_ids=None, # screen all enzymes
                    tokenizer=self.tokenizer,
                    max_length=self.train_config['max_length']
                )

                self.val_labels, _, _ = get_eval_labels(
                    eval_rxn_enz_ids=self.val_ids,
                    eval_rxn_ids=self.val_rxn_dataset.keys,
                    eval_enz_ids=self.val_enz_dataset.enz_ids,
                    train_rxn_enz_ids=self.train_ids
                )

        elif stage == 'test':
            self.test_rxn_dataset = RxnDataset(
                db_dir=self.rxn_db_dir,
                rxn_ids=self.test_ids['rxn_id'].unique(),
                spatial_pos_max=self.train_config['multi_hop_max_dist'],
                mask_unreachable=False
            )
        
            self.test_enz_dataset = EnzymeDataset(
                db_path=self.enz_db_path,
                enz_ids=None, # screen all enzymes
                tokenizer=self.tokenizer,
                max_length=self.train_config['max_length']
            )
            
            self.test_labels, rxn_id_to_idx, enz_id_to_idx = get_eval_labels(
                eval_rxn_enz_ids=self.test_ids,
                eval_rxn_ids=self.test_rxn_dataset.keys,
                eval_enz_ids=self.test_enz_dataset.enz_ids,
                train_rxn_enz_ids=self.train_ids
            )
            
            if self.test_cdts is not None:
                # after retrieval, match reactions and enzymes with cross attention
                if type(self.test_cdts) is pd.DataFrame: # ids were given
                    test_cdt_ids = self.test_cdts.copy()
                    test_cdt_ids['rxn_idx'] = test_cdt_ids['rxn_id'].map(rxn_id_to_idx)
                    test_cdt_ids['enz_idx'] = test_cdt_ids['enz_id'].map(enz_id_to_idx)
                    test_cdt_ids.sort_values('rxn_idx', inplace=True)
                    test_cdt_indices = torch.from_numpy(test_cdt_ids['enz_idx'].values).reshape(
                        self.test_labels.size(0), -1
                    )
                else: # indices were given
                    test_cdt_indices = self.test_cdts.cpu()
                    test_cdt_ids = pd.DataFrame(
                        {
                            'rxn_id': self.test_rxn_dataset.keys.repeat(test_cdt_indices.size(1)),
                            'enz_id': self.test_enz_dataset.enz_ids[test_cdt_indices.flatten().numpy()]
                        }
                    )

                self.test_dataset = RxnzymeDataset(
                    rxn_db_dir=self.rxn_db_dir,
                    enz_db_path=self.enz_db_path,
                    rxn_enz_ids=test_cdt_ids,
                    tokenizer=self.tokenizer,
                    spatial_pos_max=self.train_config['multi_hop_max_dist'],
                    mask_unreachable=False,
                    max_length=self.train_config['max_length']
                )

                self.test_labels = self.test_labels.gather(dim=1, index=test_cdt_indices)

    def train_dataloader(self):
        if self.trainer.world_size > 1:
            batch_sampler = DistributedRxnzymeSamplerForCL(
                self.train_dataset,
                batch_size=self.train_config['train_batch_size'],
                num_iters=self.train_config['num_iters'],
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size,
                shuffle=True
            )
        else:
            batch_sampler = RxnzymeSamplerForCL(
                self.train_dataset,
                batch_size=self.train_config['train_batch_size'],
                num_iters=self.train_config['num_iters'],
                shuffle=True
            )
        
        return get_dataloader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.train_config['num_workers'],
            prefetch_factor=self.train_config['prefetch_factor']
        )

    def val_dataloader(self):
        if self.val_rxn_dataset is None:
            return []
        
        batch_sampler = get_regular_sampler(
            self.val_rxn_dataset,
            batch_size=self.train_config['eval_batch_size'],
            shuffle=False,
            rank=self.trainer.global_rank,
            world_size=self.trainer.world_size
        )
        rxn_dataloader = get_dataloader(
            self.val_rxn_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.train_config['num_workers'],
            prefetch_factor=self.train_config['prefetch_factor']
        )

        batch_sampler = get_regular_sampler(
            self.val_enz_dataset,
            batch_size=self.train_config['eval_batch_size'],
            shuffle=False,
            rank=self.trainer.global_rank,
            world_size=self.trainer.world_size
        )
        enz_dataloader = get_dataloader(
            self.val_enz_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.train_config['num_workers'],
            prefetch_factor=self.train_config['prefetch_factor']
        )

        return [rxn_dataloader, enz_dataloader]

    def test_dataloader(self):
        if self.test_cdts is None:
            batch_sampler = get_regular_sampler(
                self.test_rxn_dataset,
                batch_size=self.train_config['eval_batch_size'],
                shuffle=False,
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size
            )
            rxn_dataloader = get_dataloader(
                self.test_rxn_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.train_config['num_workers'],
                prefetch_factor=self.train_config['prefetch_factor']
            )

            batch_sampler = get_regular_sampler(
                self.test_enz_dataset,
                batch_size=self.train_config['eval_batch_size'],
                shuffle=False,
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size
            )
            enz_dataloader = get_dataloader(
                self.test_enz_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.train_config['num_workers'],
                prefetch_factor=self.train_config['prefetch_factor']
            )

            return [rxn_dataloader, enz_dataloader]
        
        else:
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

class ProRxnDataModuleForLTR(LightningDataModule):
    def __init__(
            self,
            rxn_db_dir,
            enz_db_path,
            tokenizer,
            train_config,
            train_labels=None,
            val_labels=None,
            test_labels=None,
            pred_ids=None
        ):
        super().__init__()
        self.rxn_db_dir = rxn_db_dir
        self.enz_db_path = enz_db_path
        self.train_config = train_config
        self.tokenizer = tokenizer
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels
        self.pred_ids = pred_ids
    
    def setup(self, stage='fit'):
        if stage == 'fit':
            self.train_dataset = RxnzymeDatasetForLTR(
                rxn_db_dir=self.rxn_db_dir,
                enz_db_path=self.enz_db_path,
                labels=self.train_labels,
                tokenizer=self.tokenizer,
                spatial_pos_max=self.train_config['multi_hop_max_dist'],
                mask_unreachable=False,
                max_length=self.train_config['max_length']
            )
        
        if stage in {'fit', 'validate'}:
            if self.val_labels is None or len(self.val_labels) == 0:
                self.val_dataset = None
            else:
                self.val_dataset = RxnzymeDatasetForLTR(
                    rxn_db_dir=self.rxn_db_dir,
                    enz_db_path=self.enz_db_path,
                    labels=self.val_labels,
                    tokenizer=self.tokenizer,
                    spatial_pos_max=self.train_config['multi_hop_max_dist'],
                    mask_unreachable=False,
                    max_length=self.train_config['max_length']
                )
        
        elif stage == 'test':
            self.test_dataset = RxnzymeDatasetForLTR(
                rxn_db_dir=self.rxn_db_dir,
                enz_db_path=self.enz_db_path,
                labels=self.test_labels,
                tokenizer=self.tokenizer,
                spatial_pos_max=self.train_config['multi_hop_max_dist'],
                mask_unreachable=False,
                max_length=self.train_config['max_length']
            )
        
        else:
            pred_labels = self.pred_ids.assign(label=0.0)
            self.pred_dataset = RxnzymeDatasetForLTR(
                rxn_db_dir=self.rxn_db_dir,
                enz_db_path=self.enz_db_path,
                labels=pred_labels,
                tokenizer=self.tokenizer,
                spatial_pos_max=self.train_config['multi_hop_max_dist'],
                mask_unreachable=False,
                max_length=self.train_config['max_length']
            )
    
    def train_dataloader(self):
        batch_sampler = get_regular_sampler(
            self.train_dataset,
            batch_size=self.train_config['train_batch_size'],
            shuffle=True,
            rank=self.trainer.global_rank,
            world_size=self.trainer.world_size
        )
        return get_dataloader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.train_config['num_workers'],
            prefetch_factor=self.train_config['prefetch_factor']
        )
    
    def val_dataloader(self):
        if self.val_dataset is None:
            return []
        
        batch_sampler = get_regular_sampler(
            self.val_dataset,
            batch_size=self.train_config['eval_batch_size'],
            shuffle=False,
            rank=self.trainer.global_rank,
            world_size=self.trainer.world_size
        )
        return get_dataloader(
            self.val_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.train_config['num_workers'],
            prefetch_factor=self.train_config['prefetch_factor']
        )
    
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
