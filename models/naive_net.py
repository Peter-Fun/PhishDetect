import torch
import PhishDetect.models.classification_head as ch
class NaiveNet(torch.nn.Module):
    """
    A naive implementation of text classification based on 
    https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html#define-the-model
    """
    def __init__(self, cfg, vocab_size):
        """
        """
        super().__init__()
        self.cfg = cfg
        self.model_cfg = cfg.NAIVE_NET
        self.embedding = torch.nn.EmbeddingBag(vocab_size, self.model_cfg.EMBED_DIM, sparse=False)
        self.head = ch.MlpHead(self.cfg, self.model_cfg.EMBED_DIM)
        self.init_weights()


    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)

        for n, p in self.head.named_parameters():
            if isinstance(p, torch.nn.Linear):
                torch.nn.init.uniform(p.weight)
                torch.nn.init.uniform(p.bias)


    def forward(self, texts, lengths):
        """
        Args:
            texts: the input to the model (shape: B x max_L)
                B = batch size
                max_L = max length of sequence within batch
            lengths: the true length of each sequence in the batch (shape: B)
        """
        batch_size = texts.shape[0]

        # Reshaping Input.
        x = []
        offsets = []
        cum = 0
        for b in range(batch_size):
            x.append( texts[b, 0:lengths[b]] )
            offsets.append( cum )
            cum += lengths[b].item()
        x = torch.cat(x, dim=0)
        offsets = torch.tensor(offsets, dtype=int).cuda()

        # Sanity Checks
        assert len(x.shape) == 1
        assert x.shape[0] == lengths.sum()
        assert len(offsets.shape) == 1
        assert offsets.shape[0] == batch_size

        # Run the Model
        embedded = self.embedding(x, offsets)
        return self.head(embedded)