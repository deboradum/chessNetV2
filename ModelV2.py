import mlx.core as mx
import mlx.nn as nn

from datasetGen.constants import INPUT_DIM, BIN_SIZE


class FeedForward(nn.Module):
    def __init__(self, embedding_dim=64, widening_factor=4):
        super().__init__()

        dim = embedding_dim * widening_factor

        self.l1 = nn.Linear(88, dim, bias=False)
        self.l2 = nn.Linear(dim, dim, bias=False)
        self.silu = nn.SiLU()
        self.final = nn.Linear(dim, embedding_dim, bias=False)

    def __cal__(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x)
        x = self.silu(x1) * x2

        return self.final()


class AttentionBlock(nn.Module):
    def __init__(self, dims, num_heads):
        super().__init__()
        self.layerNorm1 = nn.LayerNorm(88)
        self.attention = nn.MultiHeadAttention(
            88,
            num_heads,
        )
        self.layerNorm2 = nn.LayerNorm(88)
        self.feedForward = FeedForward()

    def __cal__(self, x):
        attention_input = self.layerNorm1(x)
        att = self.attention(attention_input)
        x += att

        mlp_input = self.layerNorm2(x)
        mlp_output = self.feedForward(mlp_input)
        x += mlp_output

        return x


class ChessNet(nn.Module):
    def __init__(self, num_layers, num_heads, vocab_size, embed_size):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size)
        self.layers = []

        for _ in range(num_layers):
            self.layers.append(AttentionBlock(88, 8))

        # TODO: Get size of x here
        self.final_layer = nn.Linear(88, BIN_SIZE)
        self.softmax = nn.Softmax()

    def __call__(self, x):
        # TODO: shift enzo
        b, seq_len = x.shape
        print(x.shape)
        h = self.token_embeddings(x)
        print(h.shape)
        return
        for layer in self.layers:
            h = layer(h)

        if self.postLayerNorm:
            h = self.postLayerNorm(h)

        if not self.train:
            h = self.softmax(h)

        return h
