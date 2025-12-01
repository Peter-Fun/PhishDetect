import torch

import PhishDetect.models.classification_head as ch


class Transformer(torch.nn.Module):
    """
    A implementation of the transformer model
    """
    def __init__(self, cfg, vocab_size):
        """
        """
        super().__init__()
        self.cfg = cfg
        self.model_cfg = cfg.TRANSFORMER
        
        # Defining the parts of the model [embedding, positional encoder, encoder, fc layer]
        self.embedding = torch.nn.Embedding(vocab_size, self.model_cfg.EMBED_DIM)
        self.positional_encoder = PositionalEncoding(
            d_model=self.model_cfg.EMBED_DIM,
            dropout=self.model_cfg.DROPOUT,
            max_len=self.cfg.DATA.MAX_SEQ_LEN,
        )
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.model_cfg.EMBED_DIM, # 96
            nhead=self.model_cfg.NHEAD,
            dim_feedforward=self.model_cfg.HIDDEN_DIM, # 64
            batch_first=True,
            dropout=self.model_cfg.DROPOUT # 0.1
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, self.model_cfg.NUM_LAYERS)

        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.head = ch.MlpHead(self.cfg, self.model_cfg.EMBED_DIM)
        self.init_weights()


    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)

        for n, p in self.encoder.named_parameters():
            if isinstance(p, torch.nn.Linear):
                torch.nn.init.uniform(p.weight)
                torch.nn.init.uniform(p.bias)

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
        batch_size, max_L = texts.shape
        x = texts

        # Define Symbols
        #   HD = Hidden Dimension
        #   NC = Number of Classes
        
        # Run the Model
        x = self.embedding(x) # x: B max_L D
        x = self.positional_encoder(x) # x: B max_L D
        mask = torch.zeros((batch_size, max_L)) # mask: B max_L
        for i, l in enumerate(lengths):
            mask[i, l:] = 1
        mask = mask.to(x.device)

        x = self.encoder(x, src_key_padding_mask=mask) # x: B max_L HD
        x = torch.permute(x, (0, 2, 1)) # B HD max_L
        x = self.avg_pool(x) # x: B HD 1
        x = x.squeeze(2) # x: B HD
        return self.head(x) # x: B NC


class PositionalEncoding(torch.nn.Module):
    """Transformer Positional Encoding
    Based on this implementation: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Positional Encoding module.
        """
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)