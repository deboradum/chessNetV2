import time

import mlx.nn as nn
import mlx.core as mx
import mlx.data as dx
import mlx.optimizers as optim

# from Model import ChessNet
from ModelV2 import ChessNet

from datasetGen.factories import buildinWinsIterableFactory

# batch_size = 2048

# test_dset = (
#     dx.stream_python_iterable(buildinWinsIterableFactory("datasetGen/builtinWinsTest.db"))
#     .batch(batch_size)
# )
# valid_dset = (
#     dx.stream_python_iterable(buildinWinsIterableFactory("datasetGen/builtinWinsVal.db"))
#     .batch(batch_size)
# )


def test(model, dset, eval_fn, num_batches=-1):
    accs = []
    for i, batch in enumerate(dset):
        if i > num_batches and num_batches != -1:
            print("breaking")
            break
        X, y = batch["x"], batch["y"]
        accs.append(eval_fn(model, X, y))
    print("accs:", mx.array(accs).mean)


def train(
    model, train_dset, val_dset, optimizer, loss_fn, eval_fn, nepochs, batch_size, log_interval=10
):
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    for epoch in range(nepochs):
        train_dset = (
            dx.stream_python_iterable(
                buildinWinsIterableFactory("datasetGen/builtinWinsTrainOVERFIT.db")
            )
            .batch(batch_size)
            .shuffle(4096)
        )
        for batch in train_dset:
            X, y = mx.array(batch["x"]), mx.array(batch["y"])
            loss, grads = loss_and_grad_fn(model, X, y)
            acc = eval_fn(model, X, y)
            print("loss:", float(loss), "acc:", float(acc))
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            if float(acc) > 0.9:
                print("Converges in ", epoch, "epochs.")
                return
    print("Did not converge")
        # test(model, val_dset, eval_fn, num_batches=100)


if __name__ == "__main__":

    def loss_fn(model, X, y):
        return mx.mean(nn.losses.cross_entropy(model(X), y))

    def eval_fn(model, X, y):
        return mx.mean(mx.argmax(model(X), axis=1) == y)



optimizers = {
    "adam": [
        # 0.00001,
        0.0001,
    ],
    # "adagrad": [
    #     0.00075,
    #     0.0005,
    # ],
    # "sgd": [
    #     0.00075,
    #     0.0005,
    #     0.0005,
    #     0.001,
    #     0.001,
    #     0.005,
    #     0.01,
    #     0.05,
    #     0.1,
    # ],
}

# Loop through each optimizer and its corresponding learning rates
for optimizer_name, learning_rates in optimizers.items():
    for batch_size in [256]:
        for lr in learning_rates:
            net = ChessNet(4, 4, 128, 512)
            print(f"{optimizer_name}, {lr}, {batch_size}")

            # Choose the optimizer based on its name
            if optimizer_name == "adam":
                optimizer = optim.Adam(lr)
            elif optimizer_name == "adagrad":
                optimizer = optim.Adagrad(lr)
            elif optimizer_name == "sgd":
                optimizer = optim.SGD(lr)

            tic = time.perf_counter()
            train(net, "train_dset", "valid_dset", optimizer, loss_fn, eval_fn, 500, batch_size)
            toc = time.perf_counter()
            taken = round(toc-tic, 2)
            print("Took", taken, "seconds")
            print()
