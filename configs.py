from dataclasses import dataclass

@dataclass
class CTMConfig:
    iterations: int = 50
    d_model: int = 512
    d_input: int = 512
    memory_length: int = 15
    heads: int = 8
    n_synch_out: int = 128
    n_synch_action: int = 128
    memory_hidden_dims: int = 128
    num_classes: int = 128
    vocab_size: int = 128
    token_embed_dim: int = 512

    seed: int = 99
    log_interval: int = 100
    batch_size: int = 64
    target_batch_size: int = 1024
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
    train_dset_path: str = "dataset/train.db"
    val_dset_path: str = "dataset/val.db"
    test_dset_path: str = "dataset/test.db"
    resume_checkpoint_path: str = "none"


@dataclass
class TransformerConfig:
    num_layers: int = 8
    num_heads: int = 8
    num_classes: int = 128
    vocab_size: int = 128
    embedding_dim: int = 1024

    seed: int = 99
    log_interval: int = 100
    batch_size: int = 64
    target_batch_size: int = 1024
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
    resume_checkpoint_path: str = "none"
