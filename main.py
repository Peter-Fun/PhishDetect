

"""
This file contains code to run the PhishDetect models.
"""

# Create the yacs config object using command line argments.
import PhishDetect.utils.arguments as args
cmd_args = args.parse_args()
cfg = args.load_config(cmd_args)

# Set the GPUs to use (must be run before importing torch anywhere)
import PhishDetect.utils.basic as basic
basic.set_visible_gpus(cfg.EXECUTION.AVAILABLE_GPUS)

import os
import torch
import yaml

import common.logging as logging
from PhishDetect.tools.train import train
from PhishDetect.tools.test import test
from PhishDetect.tools.demo import demo
from PhishDetect.tools.test_llm import test_llm

if __name__=='__main__':
    """
    Run the training/testing/demo!
    """
    run_mode = cfg.EXECUTION.MODE # (train | test | demo)
    output_dir = cfg.EXECUTION.OUTPUT_DIR

    # torch.cuda.set_device(0)

    # Create the run folder structure.
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'configs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    # Save the settings used to run the file
    if run_mode.lower() == 'train':
        with open(os.path.join(output_dir, 'configs', 'config.yaml'), 'w') as file:
            file.write( cfg.dump(indent=4) )

    # Set up the logger. Using log.(info | error | warn) will save to the output dir.
    # Regular print() still works, but will not be saved.
    logging.setup_logging(output_dir=output_dir)
    log = logging.getlogger(__name__)

    # Start the run.
    log.info(f"... Starting up a {run_mode.lower()} run ...")
    log.info(f"Using configuration: \n{cfg.dump(indent=4)}")
    if run_mode.lower() == 'train':
        train(cfg)
    elif run_mode.lower() == 'test':
        test(cfg)
    elif run_mode.lower() == 'demo':
        demo(cfg)
    elif run_mode.lower() == 'test_llm':
        test_llm(cfg)
    log.info(f"... Finished ...")