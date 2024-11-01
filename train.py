import time
import yaml

import numpy as np
import mlx.nn as nn
import mlx.core as mx
import mlx.data as dx
import mlx.optimizers as optim

from Model import ChessNet

from datasetGen.factories import buildinWinsIterableFactory
from datasetGen.constants import BIN_SIZE


def test(model, dset, batch_size, eval_fn, num_batches=-1):
    accs = []
    dset = dset.batch(batch_size)
    for i, batch in enumerate(dset):
        if i > num_batches and num_batches != -1:
            break
        X, y = mx.array(batch["x"]), mx.array(batch["y"])
        accs.append(eval_fn(model, X, y))

    return np.array(accs).mean()


def init_log_file(filepath):
    with open(filepath, "w") as f:
        f.write("epoch,batch,train_loss,train_acc,test_acc\n")


def log_loss_and_acc(
    filepath, epoch, batch, avg_train_loss, avg_train_acc, avg_test_acc, time_taken
):
    print(
        f"Epoch: {epoch}, batch: {batch} | train loss: {avg_train_loss:.2f} | train acc: {avg_train_acc:.2f} | test acc: {avg_test_acc:.2f} | Took {time_taken:.2f} seconds"
    )
    with open(filepath, "a+") as f:
        f.write(f"{epoch},{batch},{avg_train_loss},{avg_train_acc}\n")


def train(
    model,
    train_dset,
    val_dset,
    optimizer,
    loss_fn,
    eval_fn,
    nepochs,
    batch_size,
    log_path,
    log_interval=10,
):
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    init_log_file(log_path)

    best_acc = 0

    for epoch in range(nepochs):
        losses = []
        accs = []

        train_dset = train_dset.batch(batch_size).shuffle(4096)

        start = time.perf_counter()
        for i, batch in enumerate(train_dset):
            X, y = mx.array(batch["x"]), mx.array(batch["y"])
            loss, grads = loss_and_grad_fn(model, X, y)
            acc = eval_fn(model, X, y)
            losses.append(loss)
            accs.append(acc)

            if i % log_interval == 0:
                test_acc = test(model, val_dset, batch_size, eval_fn, num_batches=50)
                if test_acc > best_acc:
                    model.save_weights("model.npz")
                stop = time.perf_counter()
                time_taken = round(stop - start, 2)
                log_loss_and_acc(
                    log_path,
                    epoch,
                    i,
                    round(np.array(losses).mean(), 2),
                    round(np.array(accs).mean(), 2),
                    round(test_acc, 2),
                    time_taken,
                )
                losses = []
                accs = []
                start = time.perf_counter()

            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)


def loss_fn(model, X, y):
    return mx.mean(nn.losses.cross_entropy(model(X), y))


def eval_fn(model, X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)


if __name__ == "__main__":
    mx.random.seed(101)

    with open("train.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_overfit_dset = dx.stream_python_iterable(
        buildinWinsIterableFactory("datasetGen/builtinWinsTrainOverfit.db")
    )
    train_dset = dx.stream_python_iterable(
        buildinWinsIterableFactory("datasetGen/builtinWinsTrain.db")
    )
    test_dset = dx.stream_python_iterable(
        buildinWinsIterableFactory("datasetGen/builtinWinsTest.db")
    )
    valid_dset = dx.stream_python_iterable(
        buildinWinsIterableFactory("datasetGen/builtinWinsVal.db")
    )

    # Training hyperparams
    opt = config["optimizer"]
    lr = config["learning_rate"]
    nepochs = config["nepochs"]
    batch_size = config["batch_size"]
    if opt == "adam":
        optimizer = optim.Adam(lr)
    else:
        print(f"{opt} optimizer not supported")

    # Model hyperparams
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    embedding_dim = config["emebedding_dim"]
    net = ChessNet(num_layers, num_heads, BIN_SIZE, embedding_dim)

    print(
        f"Training with {opt} optimizer, learning rate: {lr}, batch size: {batch_size} for {nepochs} epochs.\nModel has {num_layers} layers, {num_heads} heads, {embedding_dim} dimensional embeddings and {BIN_SIZE} output classes."
    )

    # Resume from existing model
    if config["resume"] != "":
        print(f"Resuming from weights: {config['resume']}")
        net.load_weights(config["resume"])

    filename = f"{opt}_{lr}_{batch_size}_{batch_size}_{num_layers}_{num_heads}_{embedding_dim}_{BIN_SIZE}.csv"

    train(
        net,
        train_dset,
        valid_dset,
        optimizer,
        loss_fn,
        eval_fn,
        nepochs,
        batch_size,
        filename,
    )
