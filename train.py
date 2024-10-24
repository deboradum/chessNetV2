

import mlx.core as mx
import mlx.data as dx

from Model import ChessNet

from datasetGen.factories import buildinWinsIterableFactory

batch_size = 1024
train_dset = (
    dx.stream_python_iterable(buildinWinsIterableFactory("datasetGen/builtinWins.db"))
    .batch(batch_size)
    .shuffle(100)
)
test_dset = (
    dx.stream_python_iterable(buildinWinsIterableFactory("datasetGen/builtinWinsTest.db"))
    .batch(batch_size)
)
valid_dset = (
    dx.stream_python_iterable(buildinWinsIterableFactory("datasetGen/builtinWinsValid.db"))
    .batch(batch_size)
)


def train(dset):
    for batch in dset:
        x, y = batch["x"], batch["y"]


def test(dset):
    for batch in dset:
        x, y = batch["x"], batch["y"]


import time
start = time.perf_counter()
train(test_dset)
taken = round(time.perf_counter()-start, 4)
print("test took", taken)




# seq_len = 86
# input_dim = 86
# chessnet = ChessNet(seq_len)
# x = mx.random.normal(([seq_len, 3, input_dim]))
# y = chessnet(x)
# print(y)
