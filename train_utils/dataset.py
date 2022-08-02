import torch
import config
from pdb import set_trace as stop
from torch.utils.data import Dataset
from train_utils.sampler import BucketRepeatUniformSampler


def get_cumulative_dataset(train_stream, index):
    ''' Cumulative method for dataset
    Args:
        train_stream: train data of all bucket images.
        index: index value of bucket
    '''

    data_set = torch.utils.data.ConcatDataset(
        [train_stream[i].dataset.train() for i in range(index+1)])
    print('length of dataset: ')
    print(len(data_set))
    train_sampler = torch.utils.data.sampler.RandomSampler(data_set, replacement=False)
    train_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=train_sampler,
        pin_memory=False,
        drop_last=True,
    )

    return train_loader


def get_bucketsample_dataset(train_stream, index, buffer_size, repeat_sample=False):
    ''' bucket sample method for dataset
    Args:
        train_stream: train data of all bucket images.
        index: index value of bucket.
        buffer_size: size of sampled buckets.
        repeat_sample(bool): whether to repeat sample.
    '''

    data_set = torch.utils.data.ConcatDataset(
        [train_stream[i].dataset.train() for i in range(index+1)])
    print('length of dataset: ')
    print(len(data_set))
    train_sampler = BucketRepeatUniformSampler(data_set, num_buckets=index+1, 
                        repeat_sample=repeat_sample, buffer_size=buffer_size)
    train_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=train_sampler,
        pin_memory=False,
        drop_last=False,
    )

    return train_loader


def get_dataset(train_stream, index):
    ''' navie method for dataset
    Args:
        train_stream: train data of all bucket images.
        index: index value of bucket
    '''
    data_set = train_stream[index].dataset.train()
    train_sampler = torch.utils.data.sampler.RandomSampler(data_set, replacement=False)
    train_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=train_sampler,
        pin_memory=False,
        drop_last=True,
    )

    return train_loader
