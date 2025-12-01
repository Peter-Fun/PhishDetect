import os
import re
import torch

import common.logging as logging
import PhishDetect.utils.constants as constants
import common.io as io

log = logging.getlogger(__name__)


def open_checkpoint_file(path: str):
    """
    Return the contents of the checkpoint file.
    """
    contents = torch.load(path)
    return contents['model_state_dict'], contents['optimizer_state_dict'], contents['epoch']


def load_checkpoint(cfg, model, optimizer=None):
    """
    Given a configuration file, load the model checkpoint
    """
    ckpt_path = cfg.MODEL.CHECKPOINT_PATH

    if ckpt_path == '' or not os.path.exists(ckpt_path):
        start_epoch = 0
        log.info(f"Initializing weights from scratch ...")

    else:
        # When a direct path is given ex: '/path/to/checkpoints/model.pt'
        if os.path.isfile(ckpt_path):
            ckpt_file = ckpt_path

        elif os.path.isdir(ckpt_path):
            
            # A path to a run dir is given, ex: '.../YYYY-MM-DD_HH-MM-SPLIT-CFG/'
            basename = os.path.basename(ckpt_path)
            if re.search(constants.RUN_DIR_PATTERN, basename):
                ckpt_folder = os.path.join(ckpt_path, 'checkpoints')
            
            # A path to a chkpt dir is given, ex: '.../YYYY-MM-DD_HH-MM-SPLIT-CFG/checkpoints'
            elif basename == 'checkpoints':
                ckpt_folder = ckpt_path

            elif basename != 'checkpoints':
                raise IOError(f'Invalid checkpoint folder {ckpt_path}')
            
            # Get the latest checkpoint in the checkpoint folder
            ckpt_files = io.find_files_in(ckpt_folder, constants.CHKPT_FILE_PATTERN)
            ckpt_file = sorted(ckpt_files)[-1]
        
        # Load the ckpt_file
        model_state, optimizer_state, state_epoch = open_checkpoint_file(ckpt_file)
        model.load_state_dict(model_state)

        # If resuming training, load the optimizer state.
        if cfg.SOLVER.RESUME and cfg.EXECUTION.MODE.lower() == 'train':
            optimizer.load_state_dict(optimizer_state)
            start_epoch = state_epoch
            log.info(f"Resumed train @ epoch {start_epoch + 1} using {ckpt_file} ...")
        else:
            start_epoch = 0
            log.info(f"Loaded pretrained weights from {ckpt_file} ...")
        
    return model, optimizer, start_epoch