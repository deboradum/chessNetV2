import os
import time
import yaml
import wandb
import argparse

import torch
import torch.nn as nn
import mlx.data as dx
import torch.optim as optim

from dataclasses import dataclass

from ModelTorch import ChessNet
from factories import iterableFactory


@dataclass
class Config:
    num_layers: int = 8
    num_heads: int = 8
    num_classes: int = 128
    vocab_size: int = 128
    embedding_dim: int = 1024

    seed: int = 99
    log_interval: int = 100
    batch_size: int = 1024
    nepochs: int = 10
    warmup_epochs: int = 1
    warmup_learning_rate: float = 1e-6
    learning_rate: float = 1.2e-3
    final_learning_rate: float = 1.2e-5
    weight_decay: float = 0.05
    gradient_clipping_norm: float = 1.0
    optimizer: str = "adamw"
    beta_1: float = 0.9
    beta_2: float = 0.95
    save_dir: str = "checkpoints"
    train_dset_path: str = "datasetGen/balanced_train.db"
    val_dset_path: str = "datasetGen/val.db"
    test_dset_path: str = "datasetGen/test.db"


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def eval_fn(preds, y):
    acc = torch.mean((torch.argmax(preds, axis=1) == y).float())
    lax_acc = torch.mean((torch.abs(torch.argmax(preds, axis=1) - y) <= 3).float())

    return acc, lax_acc


TARGET_BATCH_SIZE = 1024


def test(model, config: Config, data_path, num_batches=-1):
    done = 0
    running_test_loss = 0.0
    running_test_acc = 0.0
    running_test_lax_acc = 0.0
    with torch.no_grad():
        for i, batch in enumerate(
            dx.stream_python_iterable(
                iterableFactory(data_path, config.num_classes)
            ).batch(config.batch_size)
        ):
            if num_batches != -1 and i > num_batches:
                break
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                X, y = (
                    torch.tensor(batch["x"]).to(device),
                    torch.tensor(batch["y"]).to(device),
                )
                preds = model(X)
                loss = nn.CrossEntropyLoss()(preds, y)
                acc, lax_acc = eval_fn(preds, y)

                running_test_loss += loss
                running_test_acc += acc
                running_test_lax_acc += lax_acc
            done += 1

    return (
        running_test_loss / done,
        running_test_acc / done,
        running_test_lax_acc / done,
    )


def train(
    model,
    config: Config,
    optimizer,
    scheduler,
):
    warmup_lr = config.warmup_learning_rate
    initial_lr = config.learning_rate
    global_step = 0

    accumulation_update_interval = TARGET_BATCH_SIZE // config.batch_size
    assert (
        accumulation_update_interval >= 1
    ), f"accumulation_update_interval must be one or greater. (is {accumulation_update_interval})"
    print(
        f"Target batch size is {TARGET_BATCH_SIZE}, training batch size is {config.batch_size}. Updating model parameters every {accumulation_update_interval} steps."
    )

    for epoch in range(config.nepochs):
        if epoch < config.warmup_epochs:
            print("Warmup epoch")
            # Linear warmup
            lr = warmup_lr + (initial_lr - warmup_lr) * (epoch / config.warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        else:
            scheduler.step()

        model.train()
        start = time.perf_counter()
        running_loss = 0.0
        running_acc = 0.0
        running_lax_acc = 0.0
        current_lr = optimizer.param_groups[0]["lr"]
        for i, b in enumerate(
            dx.stream_python_iterable(
                iterableFactory(config.train_dset_path, config.num_classes)
            )
            .batch(config.batch_size)
            .shuffle(8192)
        ):
            X, y = (torch.tensor(b["x"]).to(device), torch.tensor(b["y"]).to(device))

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                preds = model(X)
                loss = nn.CrossEntropyLoss()(preds, y)
                acc, lax_acc = eval_fn(preds, y)
                running_loss += loss.item()
                loss = loss / accumulation_update_interval
            loss.backward()

            running_acc += acc
            running_lax_acc += lax_acc

            if (i + 1) % accumulation_update_interval == 0:
                if config.gradient_clipping_norm != 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clipping_norm
                    )
                optimizer.step()
                optimizer.zero_grad()

            global_step += config.batch_size

            if i and i % config.log_interval == 0:
                taken = time.perf_counter() - start
                avg_loss = running_loss / config.log_interval
                avg_acc = running_acc / config.log_interval
                avg_lax_acc = running_lax_acc / config.log_interval
                ips = config.log_interval / taken

                wandb.log(
                    {
                        "epoch": epoch,
                        "batch": i,
                        "steps": global_step,
                        "train_loss": avg_loss,
                        "train_acc": avg_acc,
                        "lax_train_acc": avg_lax_acc,
                        "learning_rate": current_lr,
                    }
                )
                print(
                    f"Epoch {epoch}, step {i} (global step {global_step}),",
                    f"Avg Loss: {avg_loss:.4f}, Avg acc: {avg_acc:.2f}, Avg lax acc: {avg_lax_acc:.2f},",
                    f"Time Taken: {taken:.2f}s, ({ips:.2f} i/s)",
                )

                running_loss = 0.0
                running_acc = 0.0
                running_lax_acc = 0.0
                start = time.perf_counter()

        # In case any gradients remain
        if (i + 1) % accumulation_update_interval != 0:
            if config.gradient_clipping_norm != 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clipping_norm
                )
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            eval_loss, eval_acc, lax_eval_acc = test(
                model, config, config.val_dset_path
            )
        wandb.log(
            {
                "epoch": epoch,
                "eval_loss": eval_loss,
                "eval_acc": eval_acc,
                "lax_eval_acc": lax_eval_acc,
            }
        )
        torch.save(model.state_dict(), f"{config.save_dir}/epoch_{epoch}")

    return test(model, config, config.test_dset_path)


def get_optimizer(config: Config, net):
    print("Preparing optimizer")
    if config.optimizer == "AdamW":
        optimizer = optim.AdamW(
            net.parameters(),
            lr=config.warmup_learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta_1, config.beta_2),
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=config.nepochs - config.warmup_epochs,
            eta_min=config.final_learning_rate,
        )
    else:
        raise NotImplementedError

    return optimizer, scheduler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
        config = Config(**config_dict)

    dset = dx.stream_python_iterable(
        iterableFactory(config.val_dset_path, config.num_classes)
    ).batch(config.batch_size)

    torch.manual_seed(config.seed)
    net = ChessNet(
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        vocab_size=config.vocab_size,
        embed_dim=config.embedding_dim,
        num_classes=config.num_classes,
        rms_norm=True,
    ).to(device)
    optimizer, scheduler = get_optimizer(config, net)

    os.makedirs(config.save_dir, exist_ok=True)

    wandb.init(project="chessNet", config=config)

    test_loss, test_acc, lax_test_acc = train(
        model=net,
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    wandb.log(
        {"test_loss": test_loss, "test_acc": test_acc, "lax_test_acc": lax_test_acc}
    )
