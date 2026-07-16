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
            assert len(bucket) >= self.batch_size
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
                if len(batch) < self.batch_size:
                    if self.drop_last:
                        break
                    batch.extend(bucket[:self.batch_size - len(batch)])
                batches.append(batch)
        
        if self.shuffle:
            random.shuffle(batches)
        yield from batches

class DistributedRxnBucketSampler(RxnBucketSampler):
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

        assert batch_size > 1
        super().__init__(
            dataset,
            num_atoms=num_atoms,
            batch_size=batch_size * world_size,
            bucket_size=bucket_size,
            drop_last=drop_last,
            shuffle=shuffle
        )
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
    
    def __iter__(self):
        if self.shuffle:
            seed = self.seed + self.epoch * (len(self.buckets) + 1)
        
        batches = []
        for j, bucket in enumerate(self.buckets):
            if self.shuffle: # shuffle the samples in each bucket
                rng = random.Random(seed + j)
                rng.shuffle(bucket)
            
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i: i + self.batch_size]
                if len(batch) < self.batch_size:
                    if self.drop_last:
                        break
                    batch.extend(bucket[:self.batch_size - len(batch)])
                batches.append(batch)
        
        if self.shuffle: # shuffle the batches
            rng = random.Random(seed + len(self.buckets))
            rng.shuffle(batches)
        
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
        shuffle=True,
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

class DistributedRxnSamplerForSCL(RxnSamplerForSCL):
    def __init__(
        self,
        dataset,
        labels, # a pd.Series that maps reaction ids to labels
        batch_size, # per rank
        rank=None,
        world_size=None,
        drop_last=False,
        shuffle=True,
        seed=42,
        cycles=1
    ):
        if world_size is None:
            world_size = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        if rank >= world_size or rank < 0:
            raise ValueError(f'Invalid rank {rank}, rank should be in the interval [0, {world_size - 1}]')

        assert batch_size > 1
        super().__init__(
            dataset=dataset,
            labels=labels,
            batch_size=batch_size * world_size,
            drop_last=drop_last,
            shuffle=shuffle,
            cycles=cycles
        )
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        for j in range(self.cycles):
            groups = list(self.groups.values())
            if self.shuffle: # shuffle the groups
                rng = random.Random(self.seed + self.epoch * self.cycles + j)
                rng.shuffle(groups)

            for i in range(0, len(groups), self.batch_size):
                batch_groups = groups[i: i + self.batch_size]
                if len(batch_groups) < self.batch_size:
                    if self.drop_last:
                        break
                    batch_groups.extend(groups[:self.batch_size - len(batch_groups)])

                batch_groups = batch_groups[self.rank:self.batch_size:self.world_size]
                batch = list(chain(*[random.sample(group, k=2) for group in batch_groups]))
                yield batch

    def set_epoch(self, epoch):
        '''
        When shuffle=True, this ensures all replicas use a different random ordering for each epoch.
        Otherwise, the next iteration of this sampler will yield the same ordering.
        '''
        self.epoch = epoch
