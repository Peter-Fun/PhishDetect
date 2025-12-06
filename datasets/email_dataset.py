import os
import torch
import pandas as pd
import numpy as np

import torchtext.data.utils as tt_ut
import torchtext.vocab as tt_vb

import PhishDetect.common.logging as logging

log = logging.getlogger(__name__)

"""
Dataset classes must implement three functions __init__, __len__, and __getitem__

Please see: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

class EmailDatasetCSV(torch.utils.data.Dataset):
    """
    Create a dataset to load the email csv file.
    """
    def __init__(self, cfg, vocab=None, split=None):
        """
        Construct the EmailDatasetCSV class. Prepare to load the email text and labels.

        Args:
            cfg: a yacs.config.CfgNode that contains the configuration file for dataset preparation.
            vocab: a torchtext.vocab.Vocab object
        """
        # Config (cfg) validation
        log.info('Checking the DATA config ...')
        assert os.path.isfile(cfg.DATA.CSV_PATH), "DATA.CSV_PATH is not a file"
        assert os.path.splitext(cfg.DATA.CSV_PATH)[1] == '.csv', "DATA.CSV_PATH is not a .csv file."

        # Class Initialization
        log.info('Initializing the EmailDatasetCSV object ...')
        self.cfg = cfg
        self.split = split # 'train' or 'test'
        self.data = self.make_dataset()
        self.tokenizer = tt_ut.get_tokenizer(cfg.DATA.TT_TOKENIZER)
        self.vocab = self.make_vocab(vocab)
        self.data = self.keep_split(self.data)
        self.split_assignments = None
        log.info(f'Created the EmailDatasetCSV object with {self.__len__()} samples.')


    def make_dataset(self) -> np.ndarray:
        """
        Loads the data stored as CSV and prepare it for loading into the model.
        """ 
        log.info(f'Reading the data csv file at {self.cfg.DATA.CSV_PATH} ...')
        
        # Read the CSV file:
        csv_data = pd.read_csv(self.cfg.DATA.CSV_PATH)
        self.texts = csv_data['Email Text'].tolist()
        self.labels = [1 if cat == 'Phishing Email' else 0 for cat in csv_data['Email Type'].tolist()]
        self.split_assignments = [1 if spl == self.split else 0 for spl in csv_data['split'].tolist()]
        assert len(self.texts) == len(self.labels)

        # Remove bad data from the ranks
        inds_to_remove = []
        for i in range(len(self.texts)):
            if self._is_bad_sample(self.texts[i], self.labels[i]):
                inds_to_remove.append(i)
        for i in sorted(inds_to_remove, reverse=True):
            del self.texts[i]
            del self.labels[i]
            del self.split_assignments[i]

        # Create a N x 2 array for the data:
        data = np.array([ self.texts, self.labels ], dtype=object).T
        
        return data


    def _is_bad_sample(self, text: str, label: int)-> bool:
        """
        Boolean function to check if this sample is "bad".
        """
        if not isinstance(text, str):
            return True
        if label not in [0, 1]:
            return True
        if len(text) < 3:
            return True
        if len(text) > self.cfg.DATA.MAX_SEQ_LEN:
            return True
        return False


    def make_vocab(self, vocab=None) -> tt_vb.Vocab:
        """
        Create the torchtext.vocab.Vocab object which converts tokenized texts into a sequence 
        of integers. Vocab will convert each word into an integer through a fixed lookup.
        """
        # if the vocab object is already created, use the vocab
        if isinstance(vocab, tt_vb.Vocab):
            log.info('Using a pre-loaded Vocabulary ...')
            return vocab
        
        # Build a vocab object from the data using an iterator
        else:
            log.info('Building a new Vocabulary using data ...')
            
            def yield_tokens():
                """An iterator over text inputs"""
                for i in range(self.__len__()):
                    yield self.tokenizer( self.data[i, 0] )

            new_vocab = tt_vb.build_vocab_from_iterator(yield_tokens(), specials=["<unk>"])
            new_vocab.set_default_index(new_vocab["<unk>"])
            
            return new_vocab

    
    def keep_split(self, data):
        """
        Remove data from wrong split assignments
        """
        # Remove bad data from the ranks
        inds_to_remove = []
        for i in range(len(self.texts)):
            if self.split_assignments[i] == 0:
                inds_to_remove.append(i)
        for i in sorted(inds_to_remove, reverse=True):
            del self.texts[i]
            del self.labels[i]
            del self.split_assignments[i]

        data = np.array([ self.texts, self.labels ], dtype=object).T

        return data


    def get_vocab(self) -> tt_vb.Vocab:
        """
        Return the vocab that is used by the model.
        """
        return self.vocab


    def __len__(self) -> int:
        """
        Return the number of data samples.
        """
        return self.data.shape[0]


    def __getitem__(self, i: int):
        """
        Obtain the training sample at index i.

        Args:
            i: the index to use to get the training sample

        Returns:
            text: the text input used to train the model. (str)
            label: the label associated with the sample. (0=not phish, 1=phish)
        """
        text, label = self.data[i,:]
        text = torch.tensor(self.vocab(self.tokenizer(text)))
        return text, label
    
    
    @staticmethod
    def collate_fn(batch):
        """
        Given a list of (text, label) tuples that form a batch, return a collated batch.
        """
        batch_size = len(batch)

        # Find the text input (tensor of ints) that is the longest.
        longest_length = 0
        for text, label in batch:
            if text.shape[0] > longest_length:
                longest_length = text.shape[0]

        # Use a tensor to contain the batch's text, padding the shorter texts
        batch_texts = torch.zeros((batch_size, longest_length), dtype=int)
        batch_lengths = torch.zeros((batch_size,), dtype=int)
        batch_labels = torch.zeros((batch_size,), dtype=int)
        for i, (sample_text, sample_label) in enumerate(batch):
            sample_size = sample_text.shape[0]
            batch_texts[i,:sample_size] = sample_text
            batch_lengths[i] = sample_size
            batch_labels[i] = sample_label

        return batch_texts, batch_labels, batch_lengths


if __name__=='__main__':
    print(f'This file was not meant to be run directly!')
    quit()