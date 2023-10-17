
import torch
from rphgnn.utils.nested_data_utils import nested_map
import numpy as np

def get_len(data):
    if isinstance(data, list):
        return get_len(data[0])
    else:
        return data.size(0)

def get_device(data):
    if isinstance(data, list):
        return get_device(data[0])
    else:
        return data.device
    

class NestedDataset(torch.utils.data.Dataset):

    def __init__(self, nested_data, device=None) -> None:
        self.nested_data = nested_data
        self.device = device

    def __getitem__(self, idx):

        def func(x):
            batch_data = x[idx]
            if self.device is not None:
                batch_data = batch_data.to(self.device)
            return batch_data
        
        batch_data = nested_map(self.nested_data, func)

        return batch_data
    
    def __len__(self):
        return np.ceil(get_len(self.nested_data)).astype(np.int32)


class NestedDataLoader(torch.utils.data.DataLoader):
    def __init__(self, nested_data, batch_size, shuffle, device) -> None:
        dataset = NestedDataset(nested_data, device)
        if shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=False)

        super().__init__(
            dataset=dataset,
            sampler=sampler,
            collate_fn=lambda batch: batch[0],
        )


    

