import torch
import config


def get_optimizer(model, opt="sgd", lr=None):
    """ optimizer.
    Args:
        model(nn.Module): model containing parameters to optimize
        opt: name of optimizer to create
        lr: initial learning rate
    """
    if opt == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.start_lr if lr is None else lr,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
            )
    elif opt == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.start_lr if lr is None else lr,
            weight_decay=config.weight_decay,
        )
    
    return optimizer
