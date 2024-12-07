import time
import yaml
import math
import argparse

import torch
import torch.nn as nn
import mlx.data as dx
import torch.optim as optim

from ModelTorch import ChessNet

from factories import iterableFactory

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# https://stackoverflow.com/a/62402574
def mean(l):  # noqa: E741
    return math.fsum(l) / len(l)


def test(model, bin_size, val_dset_path, batch_size, eval_fn, lax_eval_fn, num_batches=-1):
    accs = []
    lax_accs = []
    dset = dx.stream_python_iterable(iterableFactory(val_dset_path, bin_size)).batch(
        batch_size
    )
    for i, batch in enumerate(dset):
        if i > num_batches and num_batches != -1:
            break
        X, y = torch.tensor(batch["x"]).to(device), torch.tensor(batch["y"]).to(device)
        accs.append(eval_fn(model, X, y))
        lax_accs.append(lax_eval_fn(model, X, y)) if lax_eval_fn is not None else 0

    return mean(accs), mean(lax_accs)


def init_log_file(filepath):
    with open(filepath, "w") as f:
        f.write(
            "epoch,batch,train_loss,train_acc,train_lax_acc,eval_acc,eval_lax_acc\n"
        )


def log_loss_and_acc(
    filepath,
    epoch,
    batch,
    avg_train_loss,
    avg_train_acc,
    avg_lax_train_acc,
    avg_eval_acc,
    avg_lax_eval_acc,
    time_taken,
):
    print(
        f"Epoch: {epoch}, batch: {batch} | train loss: {avg_train_loss:.2f} | train acc: {avg_train_acc:.2f} | lax train acc: {avg_lax_train_acc:.2f} | eval acc: {avg_eval_acc:.2f} | lax eval acc: {avg_lax_eval_acc:.2f} | Took {time_taken:.2f} seconds"
    )
    # TODO: if resuming, resume batch and stuff from that
    with open(filepath, "a+") as f:
        f.write(
            f"{epoch},{batch},{avg_train_loss},{avg_train_acc},{avg_lax_train_acc},{avg_eval_acc},{avg_lax_eval_acc}\n"
        )


def train(
    model,
    bin_size,
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
    if config["resume"] == "":
        init_log_file(log_path)

    for epoch in range(nepochs):
        train_dset = (
            dx.stream_python_iterable(iterableFactory(train_dset_path, bin_size))
            .batch(batch_size)
            .shuffle(8192)
        )
        losses = []
        accs = []
        lax_accs = []

        start = time.perf_counter()
        for i, batch in enumerate(train_dset):
            model.train(True)
            X, y = (
                torch.tensor(batch["x"]).to(device),
                torch.tensor(batch["y"]).to(device),
            )
            optimizer.zero_grad()

            loss = loss_fn(model, X, y)
            losses.append(loss)

            acc = eval_fn(model, X, y)
            accs.append(acc)
            lax_acc = lax_eval_fn(model, X, y) if lax_eval_fn is not None else 0
            lax_accs.append(lax_acc)

            if i % log_interval == 0:
                model.eval()
                with torch.no_grad():
                    eval_acc, lax_eval_acc = test(
                        model,
                        bin_size,
                        val_dset_path,
                        batch_size,
                        eval_fn,
                        lax_eval_fn,
                        num_batches=5,
                    )
                stop = time.perf_counter()
                time_taken = round(stop - start, 2)
                log_loss_and_acc(
                    log_path,
                    epoch,
                    i,
                    round(mean(losses), 2),
                    round(mean(accs), 2),
                    round(mean(lax_accs), 2),
                    round(eval_acc, 2),
                    round(lax_eval_acc, 2),
                    time_taken,
                )

                losses = []
                accs = []
                lax_accs = []
                start = time.perf_counter()

            loss.backward()
            optimizer.step()

            if i % save_every == 0 and save_every != -1:
                torch.save(
                    model.state_dict(),
                    f"{save_model_path_base}_epoch_{epoch}_batch_{i}.pt",
                )


def classification_loss_fn(model, X, y):
    c = nn.CrossEntropyLoss()
    return torch.mean(c(model(X), y))


def classification_eval_fn(model, X, y):
    return torch.mean((torch.argmax(model(X), axis=1) == y).float())


def classification_k3_eval_fn(model, X, y):
    return torch.mean((torch.abs(torch.argmax(model(X), axis=1) - y) <= 3).float())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config["seed"])

    # Model parameters
    bin_size = config["bin_size"]
    vocab_size = config["vocab_size"]

    # Model hyperparams
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    embedding_dim = config["emebedding_dim"]

    # Training hyperparams
    opt = config["optimizer"]
    nepochs = config["nepochs"]
    batch_size = config["batch_size"]

    net = ChessNet(num_layers, num_heads, vocab_size, embedding_dim, bin_size).to(
        device
    )

    lr = config["learning_rate"]
    if opt == "adam":
        optimizer = optim.Adam(net.parameters(), lr)
    elif opt == "adamw":
        optimizer = optim.AdamW(net.parameters(), lr)
    elif opt == "adagrad":
        optimizer = optim.Adagrad(net.parameters(), lr)
    else:
        print(f"{opt} optimizer not supported")

    print(
        f"Training with {opt} optimizer, learning rate: {lr}, batch size: {batch_size} for {nepochs} epochs.\nModel has {num_layers} layers, {num_heads} heads, {embedding_dim} dimensional embeddings, vocab size {vocab_size}  and {bin_size} output classes."
    )

    # Resume from existing model
    if config["resume"] != "":
        print(f"Resuming from weights: {config['resume']}")
        net.load_state_dict(torch.load(config["resume"]))

    filename = f"torch_{opt}_{lr}_{batch_size}_{num_layers}_{num_heads}_{embedding_dim}_{bin_size}_{vocab_size}.csv"
    save_model_path_base = f"torch_{opt}_{lr}_{batch_size}_{num_layers}_{num_heads}_{embedding_dim}_{bin_size}_{vocab_size}"

    train(
        model=net,
        bin_size=bin_size,
        train_dset_path="datasetGen/train.db",
        val_dset_path="datasetGen/val.db",
        optimizer=optimizer,
        loss_fn=classification_loss_fn,
        eval_fn=classification_eval_fn,
        lax_eval_fn=classification_k3_eval_fn,
        nepochs=nepochs,
        batch_size=batch_size,
        save_every=config["save_every"],
        save_model_path_base=save_model_path_base,
        log_path=filename,
        log_interval=100,
    )
