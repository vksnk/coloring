import torch
import matplotlib.pyplot as plt
import argparse
import sys
import os


def plot_checkpoint(checkpoint_path, output_path=None):
    # 1. Verify file exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: The file '{checkpoint_path}' does not exist.")
        sys.exit(1)

    # 2. Load the checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    # 3. Extract the logs
    try:
        num_correct = checkpoint["num_correct_graphs_log"]
        num_mistakes = checkpoint["num_mistakes_log"]
        train_loss = checkpoint["train_loss_log"]
        val_loss = checkpoint["validation_loss_log"]
        epoch = checkpoint.get("epoch", len(train_loss))

        print(f"Successfully loaded checkpoint (Epoch {epoch}).")

    except KeyError as e:
        print(f"Error: Checkpoint is missing required key: {e}")
        sys.exit(1)

    # 4. Create the 2x2 Plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    # fig.suptitle(
    #     f"Training Metrics Analysis\nSource: {os.path.basename(checkpoint_path)}",
    #     fontsize=16,
    # )

    # Plot 1: Training Loss
    axs[0, 0].plot(train_loss, color="tab:blue", label="Train Loss")
    axs[0, 0].set_title("Training Loss Log")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    # Plot 2: Validation Loss
    axs[0, 1].plot(val_loss, color="tab:orange", label="Val Loss")
    axs[0, 1].set_title("Validation Loss Log")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()

    # Plot 3: Num Correct Graphs
    axs[1, 0].plot(num_correct, color="tab:green", label="Correct Graphs")
    axs[1, 0].set_title("Number of Correct Graphs")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Count")
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()

    # Plot 4: Num Mistakes
    axs[1, 1].plot(num_mistakes, color="tab:red", label="Mistakes")
    axs[1, 1].set_title("Number of Mistakes")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Count")
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 5. Save and/or Show
    if output_path:
        # DPI=300 ensures high resolution for papers/reports
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot metrics from a PyTorch checkpoint."
    )

    # Positional argument (Required)
    parser.add_argument(
        "checkpoint", type=str, help="Path to the .pt or .pth checkpoint file"
    )

    # Optional argument (The flag)
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to save the output image (e.g., results.png)",
    )

    args = parser.parse_args()

    plot_checkpoint(args.checkpoint, args.output)
