import torch 
from torch.utils.data import Sampler, Dataset
from typing import TypeVar, Optional, Iterator, Sized
import math
import random
import config
from pdb import set_trace as stop


class BucketRepeatUniformSampler(Sampler[int]):
    r"""Samples elements randomly in a bucket-uniform-balanced strategy.
    Args:
        data_source (Dataset): dataset to sample from
        generator (Generator): Generator used in sampling.
    """
    data_source: Sized

    def __init__(self, data_source: Sized, num_buckets=1, buffer_size=2, 
                 repeat_sample=False, generator=None) -> None:
        self.data_source = data_source
        self.num_buckets = num_buckets
        self.buffer_size = buffer_size
        self.repeat_sample = repeat_sample
        self.num_selected_samples = int(math.floor(len(self.data_source) // num_buckets))
        if num_buckets <= buffer_size and (not repeat_sample):
            self.num_selected_samples = len(self.data_source)
        else:
            self.num_selected_samples *= buffer_size
        self.generator = generator
    
    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source) // self.num_buckets
        
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator
        
        if self.num_buckets <= self.buffer_size and (not self.repeat_sample):
            indices = torch.randperm(len(self.data_source), generator=generator).tolist()
        elif self.num_buckets <= self.buffer_size and self.repeat_sample:
            cur_indices = torch.randperm(len(self.data_source), generator=generator).tolist()
            indices = cur_indices * int(math.ceil(self.num_selected_samples / len(self.data_source)))
            indices = indices[:self.num_selected_samples]
        else:
            pre_indices = torch.randperm(n * (self.num_buckets - 1), generator=generator).tolist()
            pre_indices = pre_indices[:n * (self.buffer_size - 1)]
            cur_indices = torch.randperm(n, generator=generator) + n * (self.num_buckets - 1)
            indices = pre_indices + cur_indices.tolist()
            # random.shuffle(indices)
        assert len(indices) == self.num_selected_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_selected_samples
