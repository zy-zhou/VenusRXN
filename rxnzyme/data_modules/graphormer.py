import os
import pandas as pd
from lightning import LightningDataModule
from .base import get_regular_sampler, get_dataloader
from ..data.datasets import (
    RxnDataset,
    RxnDatasetForMLM,
    RxnDatasetForCls
)
from ..data.samplers import (
    RxnBucketSampler,
    DistributedRxnBucketSampler,
    RxnSamplerForSCL,
    DistributedRxnSamplerForSCL
)

class RxnDataModuleForMLM(LightningDataModule):
    def __init__(
            self,
            db_dir,
            train_config,
            train_ids=None,
            val_ids=None
        ):
        super().__init__()
        self.db_dir = db_dir
        self.train_config = train_config
        self.train_ids = train_ids
        self.val_ids = val_ids

    def setup(self, stage='fit'):
        if stage == 'fit':
            self.metadata = pd.read_csv(
                os.path.join(self.db_dir, 'metadata.csv'),
                index_col='rxn_id'
            )
            self.train_lengths = self.metadata.loc[self.train_ids, 'num_atoms']

            self.train_dataset = RxnDatasetForMLM(
                db_dir=self.db_dir,
                rxn_ids=self.train_ids,
                mask_rate=self.train_config['mask_rate'],
                spatial_pos_max=self.train_config['multi_hop_max_dist'],
                mask_unreachable=False
            )

        if stage in {'fit', 'validate'}:
            if self.val_ids is None or len(self.val_ids) == 0:
                self.val_dataset = None
            else:
                self.val_dataset = RxnDatasetForMLM(
                    db_dir=self.db_dir,
                    rxn_ids=self.val_ids,
                    mask_rate=self.train_config['mask_rate'],
                    spatial_pos_max=self.train_config['multi_hop_max_dist'],
                    mask_unreachable=False
                )

    def train_dataloader(self):
        if self.train_config['bucket_size'] > 0:
            if self.trainer.world_size > 1:
                batch_sampler = DistributedRxnBucketSampler(
                    self.train_dataset,
                    num_atoms=self.train_lengths,
                    batch_size=self.train_config['train_batch_size'],
                    bucket_size=self.train_config['bucket_size'],
                    rank=self.trainer.global_rank,
                    world_size=self.trainer.world_size,
                    drop_last=True,
                    shuffle=True
                )
            else:
                batch_sampler = RxnBucketSampler(
                    self.train_dataset,
                    num_atoms=self.train_lengths,
                    batch_size=self.train_config['train_batch_size'],
                    bucket_size=self.train_config['bucket_size'],
                    drop_last=False,
                    shuffle=True
                )
        else:
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

class RxnDataModuleForSCL(LightningDataModule):
    def __init__(
            self,
            db_dir,
            train_config,
            train_labels=None,
            val_labels=None,
            test_labels=None
        ):
        super().__init__()
        self.db_dir = db_dir
        self.train_config = train_config
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels
    
    def setup(self, stage='fit'):
        if stage == 'fit':
            self.train_dataset = RxnDataset(
                db_dir=self.db_dir,
                rxn_ids=self.train_labels.index,
                spatial_pos_max=self.train_config['multi_hop_max_dist'],
                mask_unreachable=False
            )

        if stage in {'fit', 'validate'}:
            if self.val_labels is None or len(self.val_labels) == 0:
                self.val_dataset = None
            else:
                self.val_dataset = RxnDataset(
                    db_dir=self.db_dir,
                    rxn_ids=pd.concat([self.train_labels, self.val_labels]).index,
                    spatial_pos_max=self.train_config['multi_hop_max_dist'],
                    mask_unreachable=False
                )
        
        elif stage == 'test':
            self.test_dataset = RxnDataset(
                db_dir=self.db_dir,
                rxn_ids=pd.concat([self.train_labels, self.test_labels]).index,
                spatial_pos_max=self.train_config['multi_hop_max_dist'],
                mask_unreachable=False
            )

    def train_dataloader(self):
        if self.trainer.world_size > 1:
            batch_sampler = DistributedRxnSamplerForSCL(
                self.train_dataset,
                labels=self.train_labels,
                batch_size=self.train_config['train_batch_size'],
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size,
                drop_last=False,
                shuffle=True,
                cycles=self.train_config['cycles']
            )
        else:
            batch_sampler = RxnSamplerForSCL(
                self.train_dataset,
                labels=self.train_labels,
                batch_size=self.train_config['train_batch_size'],
                drop_last=False,
                shuffle=True,
                cycles=self.train_config['cycles']
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

class RxnDataModuleForCls(LightningDataModule):
    def __init__(
            self,
            db_dir,
            train_config,
            train_labels=None,
            val_labels=None,
            test_labels=None
        ):
        super().__init__()
        self.db_dir = db_dir
        self.train_config = train_config
        self.train_labels = train_labels
        self.val_labels = val_labels 
        self.test_labels = test_labels

    def setup(self, stage='fit'):
        if stage == 'fit':
            self.train_dataset = RxnDatasetForCls(
                db_dir=self.db_dir,
                labels=self.train_labels,
                spatial_pos_max=self.train_config['multi_hop_max_dist'],
                mask_unreachable=False
            )

        if stage in {'fit', 'validate'}:
            if self.val_labels is None or len(self.val_labels) == 0:
                self.val_dataset = None
            else:
                self.val_dataset = RxnDatasetForCls(
                    db_dir=self.db_dir,
                    labels=self.val_labels,
                    spatial_pos_max=self.train_config['multi_hop_max_dist'],
                    mask_unreachable=False
                )

        elif stage == 'test':
            self.test_dataset = RxnDatasetForCls(
                db_dir=self.db_dir,
                labels=self.test_labels,
                spatial_pos_max=self.train_config['multi_hop_max_dist'],
                mask_unreachable=False
            )

    def train_dataloader(self):
        if self.trainer.world_size > 1:
            batch_sampler = DistributedRxnSamplerForSCL(
                self.train_dataset,
                labels=self.train_labels,
                batch_size=self.train_config['train_batch_size'],
                rank=self.trainer.global_rank,
                world_size=self.trainer.world_size,
                drop_last=False,
                shuffle=True,
                cycles=self.train_config['cycles']
            )
        else:
            batch_sampler = RxnSamplerForSCL(
                self.train_dataset,
                labels=self.train_labels,
                batch_size=self.train_config['train_batch_size'],
                drop_last=False,
                shuffle=True,
                cycles=self.train_config['cycles']
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
