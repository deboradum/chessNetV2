import mlx.core as mx
import mlx.nn as nn

from datasetGen.constants import INPUT_DIM, BIN_SIZE


class LayerMLP(nn.Module):
    def __init__(self, embedding_dim=64, widening_factor=4):
        super().__init__()

        dim = embedding_dim * widening_factor

        self.l1 = nn.Linear(-1, dim, bias=False)
        self.l2 = nn.Linear(dim, dim, bias=False)
        self.silu = nn.SiLU()
        self.final = nn.Linear(dim, embedding_dim, bias=False)

    def __cal__(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x)
        x = self.silu(x1) * x2

        return self.final()


class ChessNet(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = []

        for _ in range(num_layers):
            l = {
                "layerNorm1": nn.LayerNorm(-1),
                "attentionBlock": nn,
                "layerNorm2": nn.LayerNorm(-1),
                "mlp": LayerMLP(),
            }

        # TODO: Get size of x here
        self.final_layer = nn.Linear(-1, BIN_SIZE)
        self.softmax = nn.Softmax()

    def __call__(self, x):
        # TODO: shift enzo

        for layer in self.layers:
            attention_input = layer["layerNorm1"](x)
            attention = layer["attentionBlock"](attention_input)
            x += attention

            mlp_input = layer["layerNorm2"](x)
            mlp_output = layer["mlp"](mlp_input)
            x += mlp_output

        if self.postLayerNorm:
            x = self.postLayerNorm(x)

        if not self.train:
            x = self.softmax(x)

        return x
