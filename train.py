import time
import yaml
import argparse

import numpy as np
import mlx.nn as nn
import mlx.core as mx
import mlx.data as dx
import mlx.optimizers as optim

from Model import ChessNet

from datasetGen.factories import buildinWinsIterableFactory
from datasetGen.constants import BIN_SIZE


def test(model, val_dset_path, batch_size, eval_fn, lax_eval_fn, num_batches=-1):
    accs = []
    lax_accs = []
    dset = dx.stream_python_iterable(buildinWinsIterableFactory(val_dset_path)).batch(
        batch_size
    )
    for i, batch in enumerate(dset):
        if i > num_batches and num_batches != -1:
            break
        X, y = mx.array(batch["x"]), mx.array(batch["y"])
        accs.append(eval_fn(model, X, y))
        lax_accs.append(lax_eval_fn(model, X, y))

    return np.array(accs).mean(), np.array(lax_accs).mean()


def init_log_file(filepath):
    with open(filepath, "w") as f:
        f.write("epoch,batch,train_loss,train_acc,train_lax_acc,test_acc,test_lax_acc\n")


def log_loss_and_acc(
    filepath,
    epoch,
    batch,
    avg_train_loss,
    avg_train_acc,
    avg_lax_train_acc,
    avg_test_acc,
    avg_lax_test_acc,
    time_taken,
):
    print(
        f"Epoch: {epoch}, batch: {batch} | train loss: {avg_train_loss:.2f} | train acc: {avg_train_acc:.2f} | lax train acc: {avg_lax_train_acc:.2f} | test acc: {avg_test_acc:.2f} | lax test acc: {avg_lax_test_acc:.2f} | Took {time_taken:.2f} seconds"
    )
    # TODO: if resuming, resume batch and stuff from that
    with open(filepath, "a+") as f:
        f.write(f"{epoch},{batch},{avg_train_loss},{avg_train_acc},{avg_lax_train_acc},{avg_test_acc},{avg_lax_test_acc}\n")


def train(
    model,
    train_dset_path,
    val_dset_path,
    optimizer,
    loss_fn,
    eval_fn,
    lax_eval_fn,
    nepochs,
    batch_size,
    save_every,
    save_model_path_base,
    log_path,
    log_interval=10,
):
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    if config["resume"] == "":
        init_log_file(log_path)

    for epoch in range(nepochs):
        train_dset = (
            dx.stream_python_iterable(
                buildinWinsIterableFactory(train_dset_path)
            )
            .batch(batch_size)
            .shuffle(8192)
        )
        losses = []
        accs = []
        lax_accs = []

        start = time.perf_counter()
        for i, batch in enumerate(train_dset):
            X, y = mx.array(batch["x"]), mx.array(batch["y"])
            loss, grads = loss_and_grad_fn(model, X, y)
            acc = eval_fn(model, X, y)
            lax_acc = lax_eval_fn(model, X, y)
            losses.append(loss)
            accs.append(acc)
            lax_accs.append(lax_acc)

            if i % log_interval == 0:
                test_acc, lax_test_acc = test(
                    model,
                    val_dset_path,
                    batch_size,
                    eval_fn,
                    lax_eval_fn,
                    num_batches=1,
                )
                stop = time.perf_counter()
                time_taken = round(stop - start, 2)
                log_loss_and_acc(
                    log_path,
                    epoch,
                    i,
                    round(np.array(losses).mean(), 2),
                    round(np.array(accs).mean(), 2),
                    round(np.array(lax_accs).mean(), 2),
                    round(test_acc, 2),
                    round(lax_test_acc, 2),
                    time_taken,
                )
                losses = []
                accs = []
                lax_accs = []
                start = time.perf_counter()

            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            if i % save_every == 0 and save_every != -1:
                model.save_weights(f"{save_model_path_base}_epoch_{epoch}_batch_{i}.npz")


def classification_loss_fn(model, X, y):
    return mx.mean(nn.losses.cross_entropy(model(X), y))

def regression_loss_fn(model, X, y):
    return mx.mean(nn.losses.mse_loss(model(X), y))

def classification_eval_fn(model, X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)

def classification_k3_eval_fn(model, X, y):
    return mx.mean(mx.abs(mx.argmax(model(X), axis=1) - y) <= 3)

def mae_regression_eval_fn(model, X, y):
    return mx.mean(mx.abs(model(X)-y))

def r2_regression_eval_fn(model, X, y):
    predictions = model(X)
    ss_total = mx.sum((y - mx.mean(y)) ** 2)
    ss_residual = mx.sum((y - predictions) ** 2)
    return 1 - (ss_residual / ss_total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    mx.random.seed(config["seed"])

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

    filename = f"mlx_{opt}_{lr}_{batch_size}_{num_layers}_{num_heads}_{embedding_dim}_{BIN_SIZE}.csv"
    save_model_path_base = f"mlx_{opt}_{lr}_{batch_size}_{num_layers}_{num_heads}_{embedding_dim}_{BIN_SIZE}"

    train(
        model=net,
        train_dset_path="datasetGen/builtinWinsOverfit.db",
        val_dset_path="datasetGen/builtinWinsVal.db",
        optimizer=optimizer,
        loss_fn=classification_loss_fn if BIN_SIZE > 1 else regression_loss_fn,
        eval_fn=classification_eval_fn if BIN_SIZE > 1 else r2_regression_eval_fn,
        lax_eval_fn=classification_k3_eval_fn if BIN_SIZE > 1 else None,
        nepochs=nepochs,
        batch_size=batch_size,
        save_every=config["save_every"],
        save_model_path_base=save_model_path_base,
        log_path=filename,
        log_interval=5,
    )
