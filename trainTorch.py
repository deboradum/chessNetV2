import time
import yaml
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


from ModelTorch import ChessNet
from datasetGen.dataLoaders import BuildinWinsDataset
from torch.utils.data import DataLoader
from datasetGen.constants import BIN_SIZE


def test(model, val_dset_path, batch_size, eval_fn, num_batches=-1):
    accs = []
    dataset = BuildinWinsDataset(val_dset_path, batch_size)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for i, (X, y) in enumerate(val_dataloader):
        if i > num_batches and num_batches != -1:
            break
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
        f.write(f"{epoch},{batch},{avg_train_loss},{avg_train_acc},{avg_test_acc}\n")


def train(
    model,
    train_dset_path,
    val_dset_path,
    optimizer,
    loss_fn,
    eval_fn,
    nepochs,
    batch_size,
    save_every,
    save_model_path_base,
    log_path,
    log_interval=10,
):
    init_log_file(log_path)

    dataset = BuildinWinsDataset(train_dset_path, batch_size)
    for epoch in range(nepochs):
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        losses = []
        accs = []

        start = time.perf_counter()
        for i, (X, y) in enumerate(train_dataloader):
            optimizer.zero_grad()

            outputs = model(X)
            loss = loss_fn(outputs, y)
            loss.backward()

            acc = eval_fn(model, X, y)
            losses.append(loss)
            accs.append(acc)

            if i % log_interval == 0:
                test_acc = test(model, val_dset_path, batch_size, eval_fn, num_batches=1)
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

            optimizer.step()

            if i % save_every == 0:
                torch.save(
                    model.state_dict(),
                    f"{save_model_path_base}_epoch_{epoch}_batch_{i}.pth",
                )


def classification_loss_fn(model, X, y):
    return torch.mean(nn.losses.cross_entropy(model(X), y))

def regression_loss_fn(model, X, y):
    return torch.mean(nn.losses.mse_loss(model(X), y))

def classification_eval_fn(model, X, y):
    return torch.mean(torch.argmax(model(X), dim=1) == y)

def mae_regression_eval_fn(model, X, y):
    return torch.mean(torch.abs(model(X)-y))

def r2_regression_eval_fn(model, X, y):
    predictions = model(X)
    ss_total = torch.sum((y - torch.mean(y)) ** 2)
    ss_residual = torch.sum((y - predictions) ** 2)
    return 1 - (ss_residual / ss_total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default="train.yaml")
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

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
        net.load_state_dict(config["resume"])

    filename = f"torch_{opt}_{lr}_{batch_size}_{num_layers}_{num_heads}_{embedding_dim}_{BIN_SIZE}.csv"
    save_model_path_base = f"torch_{opt}_{lr}_{batch_size}_{num_layers}_{num_heads}_{embedding_dim}_{BIN_SIZE}"

    train(
        model=net,
        train_dset_path="datasetGen/builtinWinsTrain.db",
        val_dset_path="datasetGen/builtinWinsVal.db",
        optimizer=optimizer,
        loss_fn=classification_loss_fn if BIN_SIZE > 1 else regression_loss_fn,
        eval_fn=classification_eval_fn if BIN_SIZE > 1 else r2_regression_eval_fn,
        nepochs=nepochs,
        batch_size=batch_size,
        save_every=config["save_every"],
        save_model_path_base=save_model_path_base,
        log_path=filename,
        log_interval=10,
    )
