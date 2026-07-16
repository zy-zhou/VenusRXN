from torch.utils.data import (
    RandomSampler,
    SequentialSampler,
    BatchSampler,
    DataLoader,
    get_worker_info
)
from torch.utils.data.distributed import DistributedSampler

def init_worker_lmdb(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    dataset.init_lmdb()

def get_regular_sampler(
    dataset,
    batch_size=None,
    drop_last=False,
    shuffle=False,
    rank=0,
    world_size=1
):
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=42,
            drop_last=drop_last
        )
    elif shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    
    if batch_size is not None:
        sampler = BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=drop_last
        )
    return sampler

def get_dataloader(
    dataset,
    batch_size=1,
    sampler=None,
    batch_sampler=None,
    num_workers=0,
    prefetch_factor=None
):
    has_lmdb = hasattr(dataset, 'init_lmdb')
    use_workers = num_workers > 0
    if has_lmdb and not use_workers:
        dataset.init_lmdb()
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        batch_sampler=batch_sampler,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        num_workers=num_workers,
        worker_init_fn=init_worker_lmdb if has_lmdb and use_workers else None,
        prefetch_factor=(prefetch_factor or 2) if use_workers else None,
        persistent_workers=use_workers
    )
