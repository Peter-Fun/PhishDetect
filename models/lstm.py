import torch
import PhishDetect.models.classification_head as ch

class LSTM(torch.nn.Module):
    """
    Implementing the LSTM
    """
    def __init__(self, cfg, vocab_size):
        """
        """
        super().__init__()
        self.cfg = cfg
        self.model_cfg = cfg.LSTM
        self.embedding = torch.nn.Embedding(vocab_size, self.model_cfg.EMBED_DIM)
        self.LSTM = torch.nn.LSTM(
            self.model_cfg.EMBED_DIM, 
            self.model_cfg.HIDDEN_DIM,
            self.model_cfg.NUM_LAYERS,
            bidirectional=self.model_cfg.BI_DIR,
            batch_first=True
        )
        self.head = ch.MlpHead(self.cfg, self.model_cfg.HIDDEN_DIM)
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
        x = texts

        if (lengths == 0).sum().item() > 0:
            import pdb; pdb.set_trace()

        # Run the Model
        x = self.embedding(x) # x: B max_L D
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths=lengths.cpu().numpy(), batch_first=True, enforce_sorted=False
        ) # x = packed_sequence
        x, _ = self.LSTM(x) # x: packed_sequence
        x = torch.nn.utils.rnn.unpack_sequence(x) # x: List[ B ]

        last_hs = []
        for b in range(batch_size):
            last_hs.append(x[b][-1])
        x = torch.stack(last_hs, dim=0) # x: B HD
        return self.head(x)