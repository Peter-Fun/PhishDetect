import argparse
import os

from PhishDetect.config.defaults import get_cfg

def parse_args():
    """
    Parse the command line arguments for main.py
    """
    parser = argparse.ArgumentParser(description='Run the model using the PhishDetect repo')
    parser.add_argument(
        '--cfg', type=str,
        help='path to a config file to use for this run.'
    )
    parser.add_argument(
        '--opts', default=[], nargs=argparse.REMAINDER,
        help='specify which configs to modify. look at the configs/defaults.py for details.'
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace):
    """
    Given an Namespace from argparse, load the run config
    """
    # Get the default configs
    cfg = get_cfg()

    # Merge in the config file
    if args.cfg is not None:
        assert os.path.isfile(args.cfg), f'Config file not found.'
        cfg.merge_from_file(args.cfg)

    # Merge in the command line config specifications
    assert len(args.opts) % 2 == 0, f'--opts are not specified correctly.'
    cfg.merge_from_list(args.opts)

    return cfg