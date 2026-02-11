import random
import torch.distributed as dist
from math import ceil
from itertools import chain
from torch.utils.data import Sampler

class RxnBucketSampler(Sampler):
    def __init__(
            self,
            dataset,
            num_atoms, # a pd.Series that maps reaction ids to atom numbers of the reactions
            batch_size,
            bucket_size,
            drop_last=False,
            shuffle=True
        ):
        num_atoms = num_atoms.loc[dataset.keys].reset_index(drop=True)
        # sort indices according to num_atoms obtained from file names
        self.indices = num_atoms.sort_values().index.tolist()
        assert batch_size > 1
        self.batch_size = batch_size
        self.bucket_size = bucket_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.buckets = self._create_buckets()
        self.num_batches = self._get_len()

    def _create_buckets(self):
        num_buckets = ceil(len(self.indices) / self.bucket_size)
        buckets = [[] for _ in range(num_buckets)]
        for i, index in enumerate(self.indices):
            bucket_idx = i // self.bucket_size
            buckets[bucket_idx].append(index)
        return buckets

    def _get_len(self):
        num_batches = 0
        for bucket in self.buckets:
            num_batches += len(bucket) // self.batch_size
            if len(bucket) % self.batch_size > 0 and not self.drop_last:
                num_batches += 1
        return num_batches
    
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        batches = []
        for bucket in self.buckets:
            if self.shuffle:
                random.shuffle(bucket)
            
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i: i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    break
                batches.append(batch)
        
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

class DistributedRxnBucketSampler(Sampler):
    def __init__(
            self,
            dataset,
            num_atoms, # a pd.Series that maps reaction ids to atom numbers of the reactions
            batch_size, # per rank
            bucket_size,
            rank=None,
            world_size=None,
            drop_last=False,
            shuffle=True,
            seed=42
        ):
        if world_size is None:
            world_size = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        if rank >= world_size or rank < 0:
            raise ValueError(f'Invalid rank {rank}, rank should be in the interval [0, {world_size - 1}]')

        num_atoms = num_atoms.loc[dataset.keys].reset_index(drop=True)
        # sort indices according to num_atoms obtained from file names
        self.indices = num_atoms.sort_values().index.tolist()
        assert batch_size > 1
        self.batch_size = batch_size * world_size # global batch size
        self.bucket_size = bucket_size
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.buckets = self._create_buckets()
        self.num_batches = self._get_len() # per rank
    
    def _create_buckets(self):
        num_buckets = ceil(len(self.indices) / self.bucket_size)
        buckets = [[] for _ in range(num_buckets)]
        for i, index in enumerate(self.indices):
            bucket_idx = i // self.bucket_size
            buckets[bucket_idx].append(index)
        return buckets
    
    def _get_len(self):
        num_batches = 0
        for bucket in self.buckets:
            assert len(bucket) >= self.batch_size
            num_batches += len(bucket) // self.batch_size
            if len(bucket) % self.batch_size > 0 and not self.drop_last:
                num_batches += 1
        return num_batches
    
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        if self.shuffle:
            seed = self.seed + self.epoch * (len(self.buckets) + 1)
        
        batches = []
        for j, bucket in enumerate(self.buckets):
            if self.shuffle: # shuffle the samples in each bucket
                random.seed(seed + j)
                random.shuffle(bucket)
            
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i: i + self.batch_size]
                if len(batch) < self.batch_size:
                    if self.drop_last:
                        break
                    batch.extend(bucket[:self.batch_size - len(batch)])
                batches.append(batch)
        
        assert len(batches) == self.num_batches
        if self.shuffle: # shuffle the batches
            random.seed(seed + j + 1)
            random.shuffle(batches)
        
        batches = [batch[self.rank:self.batch_size:self.world_size] for batch in batches]
        yield from batches
    
    def set_epoch(self, epoch):
        '''
        When shuffle=True, this ensures all replicas use a different random ordering for each epoch.
        Otherwise, the next iteration of this sampler will yield the same ordering.
        '''
        self.epoch = epoch

class RxnSamplerForSCL(Sampler):
    def __init__(
            self,
            dataset,
            labels, # a pd.Series that maps reaction ids to labels
            batch_size,
            drop_last=False,
            shuffle=False,
            cycles=1
        ):
        labels = labels.loc[dataset.keys]
        groups = labels.groupby(labels, sort=False).indices
        self.groups = {
            label: list(indices) if len(indices) > 1 else list(indices) * 2 \
                for label, indices in groups.items()
        }

        assert batch_size > 1 and len(self.groups) >= batch_size
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.cycles = cycles
        self.num_batches = self._get_len() # per cycle
    
    def _get_len(self):
        num_batches = len(self.groups) // self.batch_size
        if len(self.groups) % self.batch_size > 0 and not self.drop_last:
            num_batches += 1
        return num_batches

    def __len__(self):
        return self.num_batches * self.cycles

    def __iter__(self):
        for _ in range(self.cycles):
            groups = list(self.groups.values())
            if self.shuffle:
                random.shuffle(groups)

            for i in range(0, len(groups), self.batch_size):
                batch_groups = groups[i: i + self.batch_size]
                if len(batch_groups) < self.batch_size:
                    if self.drop_last:
                        break
                    batch_groups.extend(groups[:self.batch_size - len(batch_groups)])

                batch = list(chain(*[random.sample(group, k=2) for group in batch_groups]))
                yield batch

class DistributedRxnSamplerForSCL(Sampler):
    def __init__(
            self,
            dataset,
            labels, # a pd.Series that maps reaction ids to labels
            batch_size, # per rank
            rank=None,
            world_size=None,
            drop_last=False,
            shuffle=False,
            seed=42,
            cycles=1
        ):
        if world_size is None:
            world_size = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        if rank >= world_size or rank < 0:
            raise ValueError(f'Invalid rank {rank}, rank should be in the interval [0, {world_size - 1}]')

        labels = labels.loc[dataset.keys]
        groups = labels.groupby(labels, sort=False).indices
        self.groups = {
            label: list(indices) if len(indices) > 1 else list(indices) * 2 \
                for label, indices in groups.items()
        }

        assert batch_size > 1 and len(self.groups) >= batch_size
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.cycles = cycles
        self.num_batches = self._get_len() # per rank, per cycle
        self.total_batches = self.num_batches * self.world_size
    
    def _get_len(self):
        num_batches = len(self.groups) // self.batch_size
        if len(self.groups) % self.batch_size > 0 and not self.drop_last:
            num_batches += 1
        assert num_batches >= self.world_size

        num_batches = num_batches // self.world_size
        if num_batches % self.world_size > 0 and not self.drop_last:
            num_batches += 1
        return num_batches

    def __len__(self):
        return self.num_batches * self.cycles

    def __iter__(self):
        for j in range(self.cycles):
            groups = list(self.groups.values())
            if self.shuffle: # shuffle the groups
                random.seed(self.seed + self.epoch * self.cycles + j)
                random.shuffle(groups)
            
            batches = []
            for i in range(0, len(groups), self.batch_size):
                batch_groups = groups[i: i + self.batch_size]
                if len(batch_groups) < self.batch_size:
                    if self.drop_last:
                        break
                    batch_groups.extend(groups[:self.batch_size - len(batch_groups)])
                batches.append(batch_groups)
            
            if self.drop_last:
                batches = batches[:self.total_batches]
            else:
                batches += batches[1: self.total_batches - len(batches) + 1]
            
            batches = batches[self.rank:self.total_batches:self.world_size]
            batches = [
                list(chain(*[random.sample(group, k=2) for group in batch_groups])) \
                    for batch_groups in batches
            ]
            assert len(batches) == self.num_batches
            yield from batches

    def set_epoch(self, epoch):
        '''
        When shuffle=True, this ensures all replicas use a different random ordering for each epoch.
        Otherwise, the next iteration of this sampler will yield the same ordering.
        '''
        self.epoch = epoch

class RxnzymeSamplerForCL(Sampler):
    def __init__(
            self,
            dataset,
            batch_size,
            num_iters,
            shuffle=False
    ):
        self.rxn_enz_ids = dataset.rxn_enz_ids.reset_index(drop=True)
        self.rxn_enz_ids['rxn_idx'] = self.rxn_enz_ids['rxn_id'].factorize()[0]
        self.rxn_enz_ids['enz_idx'] = self.rxn_enz_ids['enz_id'].factorize()[0]
        self.rxn_groups = dict(list(self.rxn_enz_ids.groupby('rxn_idx', sort=False)['enz_idx']))
        self.enz_groups = dict(list(self.rxn_enz_ids.groupby('enz_idx', sort=False)['rxn_idx']))
        assert batch_size > 1
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.shuffle = shuffle
    
    def __len__(self):
        return self.num_iters
    
    def mask_2hops(self, center_pair, rxn_enz_ids):
        '''
        Discard the pairs that are 2 hops away from the center pair.
        '''
        mask = rxn_enz_ids['enz_idx'].isin(self.rxn_groups[center_pair['rxn_idx']])
        mask |= rxn_enz_ids['rxn_idx'].isin(self.enz_groups[center_pair['enz_idx']])
        return rxn_enz_ids[~mask]
    
    def reset_state(self):
        if self.shuffle:
            return self.rxn_enz_ids.sample(frac=1)
        else:
            return self.rxn_enz_ids.copy()
    
    def __iter__(self):
        i = 0
        rxn_enz_ids = self.reset_state()

        while i < self.num_iters:
            if len(rxn_enz_ids) == 0:
                assert i > 0
                rxn_enz_ids = self.reset_state()
            
            batch = []
            for _ in range(self.batch_size):
                sample = rxn_enz_ids.iloc[0]
                batch.append(sample.name)
                rxn_enz_ids = self.mask_2hops(sample, rxn_enz_ids)
                if len(rxn_enz_ids) == 0:
                    break
            
            if len(batch) == self.batch_size:
                yield batch
                i += 1

class DistributedRxnzymeSamplerForCL(Sampler):
    def __init__(
            self,
            dataset,
            batch_size, # per rank
            num_iters, # per rank
            rank=None,
            world_size=None,
            shuffle=False,
            seed=42
    ):
        if world_size is None:
            world_size = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        if rank >= world_size or rank < 0:
            raise ValueError(f'Invalid rank {rank}, rank should be in the interval [0, {world_size - 1}]')
        
        self.rxn_enz_ids = dataset.rxn_enz_ids.reset_index(drop=True)
        self.rxn_enz_ids['rxn_idx'] = self.rxn_enz_ids['rxn_id'].factorize()[0]
        self.rxn_enz_ids['enz_idx'] = self.rxn_enz_ids['enz_id'].factorize()[0]
        self.rxn_groups = dict(list(self.rxn_enz_ids.groupby('rxn_idx', sort=False)['enz_idx']))
        self.enz_groups = dict(list(self.rxn_enz_ids.groupby('enz_idx', sort=False)['rxn_idx']))
        assert batch_size > 1
        self.batch_size = batch_size * world_size # global batch size
        self.num_iters = num_iters
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.cycle = -1
    
    def __len__(self):
        return self.num_iters
    
    def mask_2hops(self, center_pair, rxn_enz_ids):
        '''
        Discard the pairs that are 2 hops away from the center pair.
        '''
        mask = rxn_enz_ids['enz_idx'].isin(self.rxn_groups[center_pair['rxn_idx']])
        mask |= rxn_enz_ids['rxn_idx'].isin(self.enz_groups[center_pair['enz_idx']])
        return rxn_enz_ids[~mask]
    
    def reset_state(self):
        self.cycle += 1

        if self.shuffle:
            seed = self.seed + self.epoch * self.num_iters + self.cycle
            return self.rxn_enz_ids.sample(frac=1, random_state=seed)
        else:
            return self.rxn_enz_ids.copy()
    
    def __iter__(self):
        i = 0
        rxn_enz_ids = self.reset_state()

        while i < self.num_iters:
            if len(rxn_enz_ids) == 0:
                assert i > 0
                rxn_enz_ids = self.reset_state()
            
            batch = []
            for _ in range(self.batch_size):
                sample = rxn_enz_ids.iloc[0]
                batch.append(sample.name)
                rxn_enz_ids = self.mask_2hops(sample, rxn_enz_ids)
                if len(rxn_enz_ids) == 0:
                    break
            
            if len(batch) == self.batch_size:
                batch = batch[self.rank:self.batch_size:self.world_size]
                yield batch
                i += 1

    def set_epoch(self, epoch):
        '''
        When shuffle=True, this ensures all replicas use a different random ordering for each epoch.
        Otherwise, the next iteration of this sampler will yield the same ordering.
        '''
        self.epoch = epoch
