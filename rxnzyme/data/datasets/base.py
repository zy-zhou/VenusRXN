import torch
import atexit
import lmdb
import pickle
from torch.utils.data import Dataset
from ..reaction.database import get_db_size

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
