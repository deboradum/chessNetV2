import re
import pandas as pd
import matplotlib.pyplot as plt


def plot_metrics(filenames):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    colors = [
        "blue",
        "green",
        "red",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "cyan",
        "magenta",
    ]

    max_loss = 0
    for filename in filenames:
        df = pd.read_csv(filename)
        max_loss = max(max_loss, df["train_loss"].max())

    max_loss += 0.1 * max_loss

    for i, filename in enumerate(filenames):
        pattern = (
            r"(.+)_(\d+\.\d+e?[-+]?\d*)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)(?:_(\w+))?\.csv"
        )
        match = re.match(pattern, filename)
        if match:
            _, opt, batch_size, _, num_layers, num_heads, embedding_dim, _ = match.groups()
            # label = f"Layers: {num_layers}, Heads: {num_heads}, Emb Dim: {embedding_dim}, Batch Size: {batch_size}"
            label = filename
        else:
            label = filename

        df = pd.read_csv(filename)

        # train loss
        axs[0].plot(
            df.index, df["train_loss"], label=label, color=colors[i % len(colors)]
        )
        axs[0].set_title("Train Loss")
        axs[0].set_xlabel("Batch")
        axs[0].set_ylabel("Loss")
        axs[0].set_ylim(0, max_loss)

        # train accuracy
        axs[1].plot(
            df.index,
            df["train_acc"],
            label=f"{label} (Train Acc)",
            color=colors[i % len(colors)],
        )
        axs[1].plot(
            df.index,
            df["train_lax_acc"],
            label=f"{label} (Train Lax Acc)",
            color=colors[i % len(colors)],
            linestyle="--",
        )
        axs[1].set_title("Train Accuracy")
        axs[1].set_xlabel("Batch")
        axs[1].set_ylabel("Accuracy")
        axs[1].set_ylim(0, 1)

        # eval accuracy
        axs[2].plot(
            df.index,
            df["eval_acc"],
            label=f"{label} (Eval Acc)",
            color=colors[i % len(colors)],
        )
        axs[2].plot(
            df.index,
            df["eval_lax_acc"],
            label=f"{label} (Eval Lax Acc)",
            color=colors[i % len(colors)],
            linestyle="--",
        )
        axs[2].set_title("Eval Accuracy")
        axs[2].set_xlabel("Batch")
        axs[2].set_ylabel("Accuracy")
        axs[2].set_ylim(0, 1)

        # epoch markers
        epochs = df["epoch"].unique()
        for ax in axs:
            for epoch in epochs:
                epoch_index = df[df["epoch"] == epoch].index[0]
                ax.axvline(x=epoch_index, color="black", linestyle="--", linewidth=0.5)
                ax.text(
                    epoch_index,  # x position
                    ax.get_ylim()[0]
                    - (ax.get_ylim()[1] - ax.get_ylim()[0])
                    * 0.05,  # y position slightly below the plot
                    f"{epoch}",
                    color="black",
                    fontsize=8,
                    ha="center",
                    va="top",
                )

    for ax in axs:
        ax.legend()

    plt.tight_layout()
    plt.show()


csv_files = [
    "mlx_adam_4e-05_256_4_4_1024_128_with_mask.csv",
    "mlx_adam_4e-05_256_2_4_1024_128_learned.csv",
    "mlx_adam_4e-05_256_4_4_1024_128_bos.csv",
    "mlx_adam_4e-05_256_4_4_1024_128.csv",
]

plot_metrics(csv_files)
