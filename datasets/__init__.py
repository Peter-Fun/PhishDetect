from .email_dataset import EmailDatasetCSV
# [Import new datasets here]

available_datasets = dir()

def build_dataset(cfg, split):
    """
    Return the dataset based on string name
    """
    name = cfg.DATA.DATASET
    assert name in available_datasets, f'Dataset {name} not added datasets/__init__.py'
    return eval(f'{name}(cfg, split=split)')