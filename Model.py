import mlx.core as mx
import mlx.nn as nn

from datasetGen.constants import INPUT_DIM, BIN_SIZE


class ChessNet(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        dims = 512
        self.l1 = nn.Linear(INPUT_DIM, dims)
        self.trans_decoder = nn.TransformerDecoder(
            num_layers=8,
            dims=dims,
            num_heads=8,
        )
        self.pooling = nn.AvgPool1d(seq_len)
        self.l2 = nn.Linear(dims, 1024)
        self.l3 = nn.Linear(1024, 2048)
        self.l4 = nn.Linear(2048, 2048)
        self.l5 = nn.Linear(2048, 2048)
        self.softmax = nn.Softmax()

    def __call__(self, x):
        x = self.l1(x)

        x = self.trans_decoder(x, x, None, None)

        x = mx.transpose(x, [1, 0, 2])
        x = self.pooling(x)
        x = mx.transpose(x, [1, 0, 2])
        x = mx.squeeze(x, 0)

        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)

        x = self.softmax(x)

        return x
