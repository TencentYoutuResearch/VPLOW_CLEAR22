import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
import config
import math


def make_scheduler(optimizer, num_epochs=None):
    """ LR Scheduler.
    Args:
        optimizer: name of optimizer to create
        num_epochs: num epoch for scheduler.
    """
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.scheduler_step_decay, gamma=config.scheduler_gamma
    )
    num_epochs = config.num_epochs if num_epochs is None else num_epochs
    
    return scheduler, num_epochs


def make_cosine_scheduler(optimizer, num_epochs=None, warmup_epochs=None):
    """ Cosine Scheduler, Cosine LR schedule with warmup.
    Args:
        optimizer: name of optimizer to create
        num_epochs: num epoch for scheduler.
        warmup_epochs: warmup epoch for scheduler.
    """
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=config.num_epochs if num_epochs is None else num_epochs,
        lr_min=config.min_lr,
        warmup_lr_init=config.warmup_lr,
        warmup_t=config.warmup_epochs if warmup_epochs is None else warmup_epochs,
    )
    num_epochs = scheduler.get_cycle_length() + config.cooldown_epochs
    
    return scheduler, num_epochs


def adjust_learning_rate(optimizer, epoch, index):
    """Decay the learning rate with half-cycle cosine after warmup.
    Args:
        optimizer: name of optimizer to create
        epoch: epoch of scheduler for bucket index.
        index: index value of num_epochs.
    """

    cur_lr = config.start_lr * (config.decay_ratio ** index)
    cur_epoch = config.num_epochs[index]

    if epoch < config.warmup_epochs:
        lr =  cur_lr * epoch / config.warmup_epochs 
    else:
        lr = config.min_lr + (cur_lr - config.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - config.warmup_epochs) / (cur_epoch - config.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    
    return lr
