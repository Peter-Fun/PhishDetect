import torch

def get_optimizer(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Get the optimizer specified in the config.
    """
    if cfg.SOLVER.OPTIMIZER.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.SOLVER.BASE_LR)
    else:
        raise NotImplementedError()

    return optimizer