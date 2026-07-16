import random
import torch.distributed as dist
from torch.utils.data import Sampler

class RxnzymeSamplerForCL(Sampler):
    def __init__(
        self,
        dataset,
        batch_size,
        num_iters,
        shuffle=True
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

class DistributedRxnzymeSamplerForCL(RxnzymeSamplerForCL):
    def __init__(
        self,
        dataset,
        batch_size, # per rank
        num_iters, # per rank
        rank=None,
        world_size=None,
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
            batch_size=batch_size * world_size,
            num_iters=num_iters,
            shuffle=shuffle
        )
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.cycle = -1

    def reset_state(self):
        self.cycle += 1

        if self.shuffle:
            seed = self.seed + self.epoch * self.num_iters + self.cycle
            return self.rxn_enz_ids.sample(frac=1, random_state=seed)
        else:
            return self.rxn_enz_ids.copy()

    def __iter__(self):
        for batch in super().__iter__():
            yield batch[self.rank:self.batch_size:self.world_size]

    def set_epoch(self, epoch):
        '''
        When shuffle=True, this ensures all replicas use a different random ordering for each epoch.
        Otherwise, the next iteration of this sampler will yield the same ordering.
        '''
        self.epoch = epoch

class RxnzymeSamplerForBLTR(Sampler):
    def __init__(
        self,
        dataset,
        batch_size, # number of positives
        drop_last=False,
        shuffle=True
    ):
        self.labels = dataset.labels.reset_index(drop=True)
        self.pos_indices = self.labels.index[self.labels['label'] == 1].tolist()
        assert len(self.pos_indices) > 0

        neg_labels = self.labels[self.labels['label'] == 0]
        neg_by_rxn = neg_labels.groupby('rxn_id', sort=False).indices
        neg_by_enz = neg_labels.groupby('enz_id', sort=False).indices
        self.neg_by_rxn = {rxn_id: list(indices) for rxn_id, indices in neg_by_rxn.items()}
        self.neg_by_enz = {enz_id: list(indices) for enz_id, indices in neg_by_enz.items()}

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_batches = self._get_len()

    def _get_len(self):
        num_pos = len(self.pos_indices)
        num_batches = num_pos // self.batch_size
        if num_pos % self.batch_size > 0 and not self.drop_last:
            num_batches += 1
        return num_batches

    def __len__(self):
        return self.num_batches

    def pick_neg(self, pos_idx, anchor):
        row = self.labels.iloc[pos_idx]
        if anchor == 'rxn_id':
            cdts = self.neg_by_rxn.get(row['rxn_id'])
            if not cdts:
                cdts = self.neg_by_enz.get(row['enz_id'])
        else:
            cdts = self.neg_by_enz.get(row['enz_id'])
            if not cdts:
                cdts = self.neg_by_rxn.get(row['rxn_id'])

        assert cdts, (
            f'Cannot find any negative for pos_idx={pos_idx} '
            f'(rxn_id={row["rxn_id"]}, enz_id={row["enz_id"]}).'
        )
        return random.choice(cdts)

    def __iter__(self):
        pos_indices = self.pos_indices.copy()
        if self.shuffle:
            random.shuffle(pos_indices)
        
        for i in range(0, len(pos_indices), self.batch_size):
            pos_batch = pos_indices[i: i + self.batch_size]
            if len(pos_batch) < self.batch_size and self.drop_last:
                break

            anchor = 'rxn_id' if random.random() < 0.5 else 'enz_id'
            neg_batch = [self.pick_neg(pos_idx, anchor) for pos_idx in pos_batch]
            yield pos_batch + neg_batch

class DistributedRxnzymeSamplerForBLTR(RxnzymeSamplerForBLTR):
    def __init__(
        self,
        dataset,
        batch_size, # number of positives per rank
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

        super().__init__(
            dataset,
            batch_size=batch_size * world_size,
            drop_last=drop_last,
            shuffle=shuffle
        )
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        pos_indices = self.pos_indices.copy()
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(pos_indices)
        
        for i in range(0, len(pos_indices), self.batch_size):
            pos_batch = pos_indices[i: i + self.batch_size]
            if len(pos_batch) < self.batch_size:
                if self.drop_last:
                    break
                padding_size = self.world_size - len(pos_batch) % self.world_size
                if padding_size < self.world_size:
                    pos_batch.extend(pos_indices[:padding_size])
            
            pos_batch = pos_batch[self.rank:len(pos_batch):self.world_size]
            anchor = 'rxn_id' if random.random() < 0.5 else 'enz_id'
            neg_batch = [self.pick_neg(pos_idx, anchor) for pos_idx in pos_batch]
            yield pos_batch + neg_batch
    
    def set_epoch(self, epoch):
        '''
        When shuffle=True, this ensures all replicas use a different random ordering for each epoch.
        Otherwise, the next iteration of this sampler will yield the same ordering.
        '''
        self.epoch = epoch
