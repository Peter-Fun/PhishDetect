import os
import re

# DO NOT IMPORT TORCH IN THIS FILE

from typing import List


def set_visible_gpus(gpus: List):
    """
    Set the OS environmental variables to show/hide select GPUs.
    """
    if len(gpus) > 0:
        str_gpus = [ str(gpu) for gpu in gpus ]
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=','.join( str_gpus )