import math
import torch
import torch.nn as nn


# Input Embeddings: Positional + Token embeddings


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        pe = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)
        i = torch.arange(0, d_model, 2, dtype=torch.float)

        div_term = torch.exp(i * -(torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)
        self.C = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        B, T = x.shape
        return self.pe[:, :T] + self.C(x)
