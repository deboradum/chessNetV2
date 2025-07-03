import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads

        assert (
            self.head_dim * self.num_heads == self.embedding_dim
        ), "embedding_dim must be divisible by num_heads"

        self.norm1 = (
            nn.RMSNorm(embedding_dim) if rms_norm else nn.LayerNorm(embedding_dim)
        )
        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.norm2 = (
            nn.RMSNorm(embedding_dim) if rms_norm else nn.LayerNorm(embedding_dim)
        )

        self.feedForward = FeedForward(embedding_dim, widening_factor=4)


    def forward(self, x):
        b, seq_len, _ = x.size()

        x_ln = self.norm1(x)

        q, k, v = self.qkv_proj(x_ln).chunk(3, dim=-1)

        q = q.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        x_f = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Reshape back to (batch_size, seq_len, embedding_dim)
        x_f = x_f.transpose(1, 2).contiguous().view(b, seq_len, self.embedding_dim)
        x_f = self.out_proj(x_f)

        x = x + x_f

        x_ln = self.norm2(x)
        x_f = self.feedForward(x_ln)

        x = x + x_f

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
