import os
import time
import yaml
import wandb
import shutil
import argparse

import torch
import torch.nn as nn
import mlx.data as dx
import torch.optim as optim

from configs import TransformerConfig, CTMConfig
from ModelCTM import ChessCTM
from ModelTransformer import ChessNet
from factories import iterableFactory


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Alternative method of converting the board to the model's input sequence.
# Instead of padding the fen and using it as a sequence, this method simply
# flattens the 8x8 board (and adds some info about castling, en-passant, etc).
def fen_to_board_seq(fen):
    raise NotImplementedError


def get_transformer_loss(preds, y):
    return nn.CrossEntropyLoss()(preds, y)

def get_transformer_acc(preds, y):
    acc = torch.mean((torch.argmax(preds, axis=1) == y).float())
    lax_acc = torch.mean((torch.abs(torch.argmax(preds, axis=1) - y) <= 3).float())

    return acc, lax_acc


def get_ctm_loss(predictions, certainties, targets, use_most_certain=True):
    """use_most_certain will select either the most certain point or the final point."""
    losses = nn.CrossEntropyLoss(reduction='none')(
        predictions.float(),
        torch.repeat_interleave(targets.unsqueeze(-1), predictions.size(-1), -1)
    )

    loss_index_1 = losses.argmin(dim=1)
    loss_index_2 = certainties[:, 1].argmax(-1)
    if not use_most_certain:
        loss_index_2[:] = -1

    batch_indexer = torch.arange(predictions.size(0), device=predictions.device)
    loss_minimum_ce = losses[batch_indexer, loss_index_1].mean()
    loss_selected = losses[batch_indexer, loss_index_2].mean()

    loss = (loss_minimum_ce + loss_selected) / 2

    return loss, loss_index_2

def get_ctm_acc(predictions, targets, where_most_certain):
    """Calculate the accuracy based on the prediction at the most certain internal tick."""
    B = predictions.size(0)
    device = predictions.device

    predictions_at_most_certain_internal_tick = predictions.argmax(1)[torch.arange(B, device=device), where_most_certain]
    acc = (predictions_at_most_certain_internal_tick == targets).float().mean().item()
    lax_acc = (torch.abs(predictions_at_most_certain_internal_tick - targets) <= 1).float().mean().item()

    return acc, lax_acc


def test(model, model_type, config: TransformerConfig|CTMConfig, data_path, num_batches=-1):
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
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                X, y = (
                    torch.tensor(batch["x"]).to(device),
                    torch.tensor(batch["y"]).to(device),
                )
                if model_type == "transformer":
                    preds = model(X)
                    loss = get_transformer_loss(preds, y)
                    acc, lax_acc = get_transformer_acc(preds, y)
                elif model_type == "ctm":
                    preds, certainties = model(X, track=False)
                    loss, where_most_certain = get_ctm_loss(preds, certainties, y)
                    acc, lax_acc = get_ctm_acc(preds, y, where_most_certain)
                else:
                    raise NotImplementedError

                running_test_loss += loss
                running_test_acc += acc
                running_test_lax_acc += lax_acc
            done += 1

    return (
        running_test_loss / done,
        running_test_acc / done,
        running_test_lax_acc / done,
    )


def clamp_decay_params(module, _input):
        with torch.no_grad():
            module.decay_params_action.data.clamp_(0, 15)
            module.decay_params_out.data.clamp_(0, 15)


def train(
    model,
    model_type,
    config: TransformerConfig|CTMConfig,
    optimizer,
    scheduler,
):
    warmup_lr = config.warmup_learning_rate
    initial_lr = config.learning_rate
    global_step = 0

    accumulation_update_interval = config.target_batch_size // config.batch_size
    assert (
        accumulation_update_interval >= 1
    ), f"accumulation_update_interval must be one or greater. (is {accumulation_update_interval})"
    print(
        f"Target batch size is {config.target_batch_size}, training batch size is {config.batch_size}. Updating model parameters every {accumulation_update_interval} steps."
    )

    for epoch in range(config.nepochs):
        if epoch < config.warmup_epochs:
            print("Warmup epoch")
            # Linear warmup
            lr = warmup_lr + (initial_lr - warmup_lr) * (epoch / config.warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
        else:
            # If no warmup is used, do not call scheduler.step before training the first epoch.
            if epoch:
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

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                if model_type == "transformer":
                    preds = model(X)
                    loss = get_transformer_loss(preds, y)
                    acc, lax_acc = get_transformer_acc(preds, y)
                elif model_type == "ctm":
                    preds, certainties = model(X, track=False)
                    loss, where_most_certain = get_ctm_loss(preds, certainties, y)
                    acc, lax_acc = get_ctm_acc(preds, y, where_most_certain)
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
            # Perform eval and save model every 1000 logging intervals
            if i % (config.log_interval * 1000) == 0:
                print("Evaluating")
                start = time.perf_counter()
                model.eval()
                eval_loss, eval_acc, lax_eval_acc = test(
                    model, model_type, config, config.val_dset_path, num_batches=1000
                )
                taken = time.perf_counter() - start
                wandb.log(
                    {
                        "steps": global_step,
                        "eval_loss": eval_loss,
                        "eval_acc": eval_acc,
                        "lax_eval_acc": lax_eval_acc,
                    }
                )
                print(
                    f"[Eval] Epoch {epoch}, step {i} (global step {global_step}),",
                    f"Eval Loss: {eval_loss:.4f}, Eval acc: {eval_acc:.2f}, Lax eval acc: {lax_eval_acc:.2f}",
                    f"Time Taken: {taken:.2f}s",
                )
                torch.save(model.state_dict(), f"{config.save_dir}/epoch_{epoch}_batch_{i}.pt")
                model.train()
                start = time.perf_counter()

        # In case any gradients remain
        if (i + 1) % accumulation_update_interval != 0:
            if config.gradient_clipping_norm != 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clipping_norm
                )
            optimizer.step()
            optimizer.zero_grad()

    return test(model, model_type, config, config.test_dset_path, num_batches=10000)


def get_optimizer(config: TransformerConfig|CTMConfig, net):
    print("Preparing optimizer")
    if config.optimizer == "AdamW":
        optimizer = optim.AdamW(
            net.parameters(),
            lr=config.warmup_learning_rate if config.warmup_epochs else config.learning_rate,
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


def get_model(model_type, config:TransformerConfig|CTMConfig):
    net:ChessNet|ChessCTM
    if model_type == "transformer":
        if not isinstance(config, TransformerConfig):
            raise TypeError("Expected TransformerConfig for transformer model")
        net = ChessNet(
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            vocab_size=config.vocab_size,
            embed_dim=config.embedding_dim,
            num_classes=config.num_classes,
            rms_norm=True,
        ).to(device)
    elif model_type == "ctm":
        if not isinstance(config, CTMConfig):
            raise TypeError("Expected CTMConfig for ctm model")
        net = ChessCTM(
            iterations=config.iterations,
            d_model=config.d_model,
            d_input=config.d_input,
            memory_length=config.memory_length,
            heads=config.heads,
            n_synch_out=config.n_synch_out,
            n_synch_action=config.n_synch_action,
            num_classes=config.num_classes,
            memory_hidden_dims=config.memory_hidden_dims,
            vocab_size=config.vocab_size,
            token_embed_dim=config.token_embed_dim,
        ).to(device)
    else:
        raise NotImplementedError

    if os.path.isfile(config.resume_checkpoint_path):
        print(f"Resuming from pre-trained weights {config.resume_checkpoint_path}")
        net.load_state_dict(
            torch.load(
                config.resume_checkpoint_path, map_location=device, weights_only=True
            )
        )

    return net


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["transformer", "ctm"])
    parser.add_argument("config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config:TransformerConfig|CTMConfig

        config_dict = yaml.safe_load(f)
        if args.model == "transformer":
            config = TransformerConfig(**config_dict)
            config_dict["model_type"] = "transformer"
        elif args.model == "ctm":
            config = CTMConfig(**config_dict)
            config_dict["model_type"] = "ctm"
        else:
            raise NotImplementedError()
        config_dict["device"] = device

    torch.manual_seed(config.seed)

    net = get_model(args.model, config)

    # Run the model with one dummy batch to initialize lazy layers.
    dummy_batch = next(iter(dx.stream_python_iterable(iterableFactory(config.train_dset_path, config.num_classes)).batch(config.batch_size)))
    dummy_input = torch.tensor(dummy_batch["x"]).to(device)
    with torch.no_grad():
        _ = net(dummy_input)

    print("Compiling network")
    net = torch.compile(net)

    if args.model == "ctm":
        net.register_forward_pre_hook(clamp_decay_params) # type: ignore

    num_params = sum(p.numel() for p in net.parameters()) # type: ignore
    print(f"Model parameters {num_params:,}")
    config_dict["num_params"] = num_params

    optimizer, scheduler = get_optimizer(config, net)

    os.makedirs(config.save_dir, exist_ok=True)
    shutil.copy(args.config, config.save_dir)

    wandb.init(project="chessNet", config=config_dict)
    print("Training on device:", device)
    test_loss, test_acc, lax_test_acc = train(
        model=net,
        model_type=args.model,
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    wandb.log({"test_loss": test_loss, "test_acc": test_acc, "lax_test_acc": lax_test_acc})
