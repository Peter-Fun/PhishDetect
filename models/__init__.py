from .naive_net import NaiveNet
from .gated_recurrent_unit_net import GatedRecurrentUnit
from .transformer import Transformer
from .lstm import LSTM
available_models = dir()

import torch

def get_model(cfg, *args, **kwargs):
    """
    Return the model based on string name
    """
    name = cfg.MODEL.ARCHITECTURE
    assert name in available_models, f'Model {name} not added models/__init__.py'
    return eval(f'{name}')

def count_params(model: torch.nn.Module) -> int:
    """
    Get the number of trainable parameters for a given  (torch.nn.Module)
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)