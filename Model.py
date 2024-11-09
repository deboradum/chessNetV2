import mlx.core as mx
import mlx.nn as nn

from datasetGen.constants import BIN_SIZE


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
    def __init__(self, embedding_dim, num_heads):
        super().__init__()
        self.layerNorm1 = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiHeadAttention(
            embedding_dim,
            num_heads,
        )
        self.layerNorm2 = nn.LayerNorm(embedding_dim)
        self.feedForward = FeedForward(embedding_dim)

    def __call__(self, x):
        attention_input = self.layerNorm1(x)
        att = self.attention(attention_input, attention_input, attention_input)
        x += att

        mlp_input = self.layerNorm2(x)
        mlp_output = self.feedForward(mlp_input)
        x += mlp_output

        return x


class ChessNet(nn.Module):
    def __init__(self, num_layers, num_heads, vocab_size, embed_dim, max_seq_len=86):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.positional_embeddings = nn.Embedding(max_seq_len, embed_dim)

        self.layers = []
        for _ in range(num_layers):
            self.layers.append(TransformerBlock(embed_dim, 8))

        self.final_layer = nn.Linear(embed_dim, BIN_SIZE)

    def __call__(self, x):
        # TODO: shift enzo
        b, seq_len = x.shape

        h = self.token_embeddings(x)
        positions = mx.arange(seq_len).reshape(1, seq_len)  # Create a 1D array of shape (1, seq_len)
        positions = mx.tile(positions, (b, 1))  # Repeat the array b times to create shape (b, seq_len)
        h += self.positional_embeddings(positions)

        for layer in self.layers:
            h = layer(h)

        logits = self.final_layer(h)

        return logits[:, -1, :]  # Only check logits for last token
