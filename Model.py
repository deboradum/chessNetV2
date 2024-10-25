import mlx.core as mx
import mlx.nn as nn

from datasetGen.constants import INPUT_DIM, BIN_SIZE


class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        dims = 512
        self.l1 = nn.Linear(INPUT_DIM, dims)
        self.relu1 = nn.ReLU()

        self.trans_decoder = nn.TransformerDecoder(
            num_layers=8,
            dims=dims,
            num_heads=8,
        )
        # self.pooling = nn.AvgPool1d(1)
        self.l2 = nn.Linear(dims, 2048)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(2048, 512)
        self.relu3 = nn.ReLU()

        self.trans_decoder2 = nn.TransformerDecoder(
            num_layers=8,
            dims=512,
            num_heads=8,
        )

        self.l4 = nn.Linear(512, 4096)
        self.relu4 = nn.ReLU()
        self.l5 = nn.Linear(4096, BIN_SIZE)

        self.softmax = nn.Softmax()

    def __call__(self, x):
        x = mx.squeeze(x, 0)
        x = self.relu1(self.l1(x))

        x = mx.expand_dims(x, 0)
        x = self.trans_decoder(x, x, None, None)
        x = mx.squeeze(x, 0)

        x = self.relu2(self.l2(x))
        x = self.relu3(self.l3(x))

        x = mx.expand_dims(x, 0)
        x = self.trans_decoder2(x, x, None, None)
        x = mx.squeeze(x, 0)

        x = self.relu4(self.l4(x))
        x = self.l5(x)

        # x = self.softmax(x)

        return x
