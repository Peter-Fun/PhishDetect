import torch

def get_loss_func(cfg) -> torch.nn.Module:
    """Get the loss function specified in the config.
    """
    if cfg.SOLVER.LOSS_FN.lower() == 'crossentropyloss':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError()

    return criterion