import math
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, widening_factor=4):
        super().__init__()

        dim = embedding_dim * widening_factor

        self.l1 = nn.Linear(embedding_dim, dim, bias=False)
        self.l2 = nn.Linear(embedding_dim, dim, bias=False)
        self.silu = nn.SiLU()
        self.final = nn.Linear(dim, embedding_dim, bias=False)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x)
        x = self.silu(x1) * x2

        return self.final(x)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, rms_norm=True):
        super().__init__()
        self.norm1 = (
            nn.RMSNorm(embedding_dim) if rms_norm else nn.LayerNorm(embedding_dim)
        )
        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, batch_first=True
        )
        self.norm2 = (
            nn.RMSNorm(embedding_dim) if rms_norm else nn.LayerNorm(embedding_dim)
        )

        self.feedForward = FeedForward(embedding_dim, widening_factor=4)


    def forward(self, x):
        b, seq_len, _ = x.size()

        x_ln = self.norm1(x)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        )
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float("-inf"))
        x_f, _ = self.attention(x_ln, x_ln, x_ln, attn_mask=causal_mask)

        x = x + x_f

        x_ln = self.norm2(x)
        x_f = self.feedForward(x_ln)

        x = x * x_f

        return x


class ChessNet(nn.Module):
    def __init__(self, num_layers, num_heads, vocab_size, embed_dim, num_classes, max_seq_len=87, rms_norm=True):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.positional_embeddings = nn.Embedding(max_seq_len, embed_dim)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, rms_norm)
                for _ in range(num_layers)
            ]
        )

        self.final_layer = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        b, seq_len = x.shape

        # Padding with BOS token
        bos_array = torch.zeros((b, 1), dtype=torch.int64).to(x.device)
        x = torch.cat([bos_array, x], axis=1)

        b, seq_len = x.shape

        h = self.token_embeddings(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(b, -1)  # (b, seq_len)
        h = h + self.positional_embeddings(positions)

        for layer in self.layers:
            h = layer(h)

        logits = self.final_layer(h)

        return logits[:, -1, :]  # Only check logits for last token
