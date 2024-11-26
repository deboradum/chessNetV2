import mlx.core as mx
import mlx.nn as nn


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, widening_factor=4):
        super().__init__()

        dim = embedding_dim * widening_factor

        self.l1 = nn.Linear(embedding_dim, dim, bias=False)
        self.l2 = nn.Linear(embedding_dim, dim, bias=False)
        self.silu = nn.SiLU()
        self.final = nn.Linear(dim, embedding_dim, bias=False)

    def __call__(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x)
        x = self.silu(x1) * x2

        return self.final(x)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, rms_norm=False):
        super().__init__()
        self.norm1 = (
            nn.RMSNorm(embedding_dim) if rms_norm else nn.LayerNorm(embedding_dim)
        )
        self.attention = nn.MultiHeadAttention(
            embedding_dim,
            num_heads,
        )
        self.norm2 = (
            nn.RMSNorm(embedding_dim) if rms_norm else nn.LayerNorm(embedding_dim)
        )

        # TODO: tweak widening factor
        self.feedForward = FeedForward(embedding_dim, widening_factor=4)

    def __call__(self, x):
        x_ln = self.norm1(x)
        x_f = self.attention(
            x_ln,
            x_ln,
            x_ln,
            mask=self.attention.create_additive_causal_mask(x.shape[1]),
        )
        x = x + x_f

        x_ln = self.norm2(x)
        x_f = self.feedForward(x_ln)
        x = x + x_f

        return x


class ChessNet(nn.Module):
    def __init__(self, num_layers, num_heads, vocab_size, embed_dim, num_classes, max_seq_len=87, rms_norm=False):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.positional_embeddings = nn.Embedding(max_seq_len, embed_dim)

        self.layers = []
        for _ in range(num_layers):
            self.layers.append(TransformerBlock(embed_dim, num_heads))

        self.final_layer = nn.Linear(embed_dim, num_classes)

    def __call__(self, x):
        b, seq_len = x.shape

        # Padding with BOS token
        bos_array = mx.zeros((b, 1), dtype=mx.int64)
        x = mx.concatenate([bos_array, x], axis=1)

        b, seq_len = x.shape

        h = self.token_embeddings(x)
        positions = mx.arange(seq_len).reshape(1, seq_len)  # (1, seq_len)
        positions = mx.tile(positions, (b, 1))  # (b, seq_len)
        h = h + self.positional_embeddings(positions)

        for layer in self.layers:
            h = layer(h)

        logits = self.final_layer(h)

        return logits[:, -1, :]  # Only check logits for last token
