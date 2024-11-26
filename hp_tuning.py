import time
import math
import optuna

import mlx.nn as nn
import mlx.core as mx
import mlx.data as dx
import mlx.optimizers as optim

from Model import ChessNet

from datasetGen.factories import iterableFactory


# https://stackoverflow.com/a/62402574
def mean(l):  # noqa: E741
    return math.fsum(l) / len(l)


def test(
    model, bin_size, val_dset_path, batch_size, eval_fn, lax_eval_fn, num_batches=-1
):
    accs = []
    lax_accs = []
    dset = dx.stream_python_iterable(iterableFactory(val_dset_path, bin_size)).batch(
        batch_size
    )
    for i, batch in enumerate(dset):
        if i > num_batches and num_batches != -1:
            break
        X, y = mx.array(batch["x"]), mx.array(batch["y"])
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
    avg_test_acc,
    avg_lax_test_acc,
    time_taken,
):
    print(
        f"Epoch: {epoch}, batch: {batch} | train loss: {avg_train_loss:.2f} | train acc: {avg_train_acc:.2f} | lax train acc: {avg_lax_train_acc:.2f} | test acc: {avg_test_acc:.2f} | lax test acc: {avg_lax_test_acc:.2f} | Took {time_taken:.2f} seconds"
    )
    # TODO: if resuming, resume batch and stuff from that
    with open(filepath, "a+") as f:
        f.write(
            f"{epoch},{batch},{avg_train_loss},{avg_train_acc},{avg_lax_train_acc},{avg_test_acc},{avg_lax_test_acc}\n"
        )


def evaluate_model(
    model,
    bin_size,
    train_dset_path,
    val_dset_path,
    test_dset_path,
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
    init_log_file(log_path)

    best_acc = 0
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
            X, y = mx.array(batch["x"]), mx.array(batch["y"])
            loss, grads = loss_and_grad_fn(model, X, y)
            losses.append(loss)

            acc = eval_fn(model, X, y)
            accs.append(acc)
            lax_acc = lax_eval_fn(model, X, y) if lax_eval_fn is not None else 0
            lax_accs.append(lax_acc)

            if i % log_interval == 0:
                eval_acc, lax_eval_acc = test(
                    model,
                    bin_size,
                    val_dset_path,
                    batch_size,
                    eval_fn,
                    lax_eval_fn,
                    num_batches=10,
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

                if eval_acc > best_acc:
                    best_acc = eval_acc
                    model.save_weights("best.npz")

                losses = []
                accs = []
                lax_accs = []
                start = time.perf_counter()

            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

    # Get test acc for the best model
    model.load_weights("best.npz")
    final_test_acc, _ = test(
        model,
        val_dset_path,
        batch_size,
        eval_fn,
        lax_eval_fn,
        num_batches=500,
    )
    return final_test_acc


def classification_loss_fn(model, X, y):
    return mx.mean(nn.losses.cross_entropy(model(X), y))


def classification_eval_fn(model, X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)


def classification_k3_eval_fn(model, X, y):
    return mx.mean(mx.abs(mx.argmax(model(X), axis=1) - y) <= 3)

# ---------------------------


def objective(trial):
    mx.random.seed(1)

    bin_size = 128
    vocab_size = 128

    embedding_dim = 1024
    batch_size = 4
    num_layers = 4
    num_heads = 4

    lr = trial.suggest_float("learning_rate", 2e-5, 8e-5, log=True)
    rms_norm = trial.suggest_categorical("rms_norm", [True, False])
    opt = trial.suggest_categorical("optimizer", ["adam", "adamw"])

    if opt == "adam":
        optimizer = optim.Adam(lr)
    elif opt == "adamw":
        optimizer = optim.AdamW(lr)
    elif opt == "adagrad":
        optimizer = optim.Adagrad(lr)

    filename = f"mlx_{opt}_{lr}_{batch_size}_{num_layers}_{num_heads}_{embedding_dim}_{bin_size}_{rms_norm}.csv"
    save_model_path_base = f"mlx_{opt}_{lr}_{batch_size}_{num_layers}_{num_heads}_{embedding_dim}_{bin_size}"

    train_dset_path = "datasetGen/hptune.db"
    val_dset_path = "datasetGen/val.db"
    test_dset_path = "datasetGen/test.db"

    model = ChessNet(num_layers, num_heads, vocab_size, embedding_dim, bin_size, rms_norm=rms_norm)
    acc = evaluate_model(
        model,
        bin_size,
        train_dset_path,
        val_dset_path,
        test_dset_path,
        optimizer,
        classification_loss_fn,
        classification_eval_fn,
        classification_k3_eval_fn,
        2,
        batch_size,
        -1,
        save_model_path_base,
        filename,
        log_interval=10,
    )

    return acc


if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=20)

    print("Best Hyperparameters:", study.best_params)
    print("Best Accuracy:", study.best_value)
