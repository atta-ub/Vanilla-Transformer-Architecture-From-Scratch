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


# mult head attention with optional causal attention


class AttentionLayer(nn.Module):
    def __init__(self, n_heads, d_model, max_seq_len, use_causal_mask=False):
        assert d_model % n_heads == 0
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model

        self.W_K = nn.Linear(d_model, d_model)
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.W_O = nn.Linear(d_model, d_model)

        self.use_causal_mask = use_causal_mask

        # mask [1, 1, max_seq_len, max_seq_len]

        torch_ones = torch.ones(max_seq_len, max_seq_len)
        mask = torch.tril(torch_ones).view(1, 1, max_seq_len, max_seq_len)

        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        B, T, _ = x.shape

        K = self.W_K(x)
        Q = self.W_Q(x)
        V = self.W_V(x)

        # reshape and transpose to separate heads
        # Target shape: [batch_size, num_heads, seq_len, d_k]

        K = K.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        Q = Q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention for Q and K^T
        # K.transpose(-2, -1) shapes K to [batch_size, num_heads, d_k, seq_len]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # attention mask
        if self.use_causal_mask:
            scores = scores.masked_fill(
                self.causal_mask[:, :, :T, :T] == 0, float("-inf")
            )

        # scores to probabilities
        probs = torch.softmax(scores, dim=-1)
        context_vector = torch.matmul(probs, V)

        # concatenate heads back together: [batch_size, seq_len, num_heads, d_k]
        # flatten the last two dimensions back into d_model
        out = context_vector.transpose(1, 2).reshape(B, T, self.d_model)

        out = self.W_O(out)
        return out


# cross multi-head attention:


class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_model):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x, enc_output):
        # x: decoder hidden states [B, T_tgt, d_model]
        # enc_output: encoder final states [B, T_src, d_model]
        B, T_tgt, _ = x.shape
        _, T_src, _ = enc_output.shape

        # queries come from the target (decoder context)
        Q = self.W_Q(x).view(B, T_tgt, self.n_heads, self.head_dim).transpose(1, 2)

        # keys & values come from the source (encoder memory)
        K = (
            self.W_K(enc_output)
            .view(B, T_src, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.W_V(enc_output)
            .view(B, T_src, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        # shape of scores: [B, n_heads, T_tgt, T_src]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # cross-attention does not use a causal mask
        probs = torch.softmax(scores, dim=-1)

        out = torch.matmul(probs, V).transpose(1, 2).reshape(B, T_tgt, self.d_model)
        return self.W_O(out)


# feedforward network
class FFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.d_ff = 4 * d_model

        self.l1 = nn.Linear(d_model, 4 * d_model)
        self.l2 = nn.Linear(4 * d_model, d_model)

        self.activate = nn.ReLU()

    def forward(self, mha_out):
        B, T, _ = mha_out.shape
        l1_o = self.l1(mha_out)
        l1_a = self.activate(l1_o)
        l2_o = self.l2(l1_a)

        return l2_o


#  layer norm


class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        self.gamma = nn.Parameter(torch.ones(self.d_model))
        self.beta = nn.Parameter(torch.zeros(self.d_model))

    def forward(self, in1p2, epsilon=1e-5):

        # mean and var for each token across the d_model dimension
        mu = in1p2.mean(axis=-1, keepdims=True)
        var = in1p2.var(axis=-1, unbiased=False, keepdims=True)

        o_normalized = (in1p2 - mu) / torch.sqrt(var + epsilon)

        # scale and shift using the learnable parameters
        output = (o_normalized * self.gamma) + self.beta

        return output
