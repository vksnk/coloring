from loss import potts_loss, entropy_loss
from model import GCCN
from dataset import RigSetDataset
from evaluate import evaluate_dataset

import os

import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader


CHECKPOINT_NAME = "checkpoints/checkpoint.pth"
BEST_CHECKPOINT_NAME = "checkpoints/best_checkpoint.pth"

if __name__ == "__main__":
    if torch.mps.is_available():
        # MPS seems to be much slower for this task.
        print("Using cpu backend...")
        device = torch.device("cpu")
        pin_memory = False
    elif torch.cuda.is_available():
        print("Using cuda backend...")
        device = torch.device("cuda")
        pin_memory = True
    else:
        print("Using cpu backend...")
        device = torch.device("cpu")
        pin_memory = False

    dataset = RigSetDataset("data/")
    train_dataset = dataset[dataset.train_mask]
    val_dataset = dataset[dataset.val_mask]

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=pin_memory
    )

    model = GCCN(dataset.num_node_features, dataset.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,  # Number of epochs for the first restart
        T_mult=1,  # Double the cycle length after every restart (50, then 100, then 200...)
        eta_min=1e-6,  # Minimum LR to reach at the bottom of the curve
    )

    if os.path.exists(CHECKPOINT_NAME):
        checkpoint = torch.load(CHECKPOINT_NAME, weights_only=True, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        print(f"Loaded saved checkpoint from epoch #{start_epoch - 1}")
    else:
        start_epoch = 0
        best_loss = float("inf")

    model.train()

    for epoch in range(start_epoch, 400):
        total_loss = 0.0
        out = None
        # Iterate over batches.
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)

            # torch.set_printoptions(profile="full")
            # print(F.softmax(out, dim=1)[0:25])
            # torch.set_printoptions(profile="default")

            loss = potts_loss(out, batch.edge_index) + 0.02 * entropy_loss(out)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        num_correct, val_loss = evaluate_dataset(model, val_loader, device)

        print(
            f"Epoch #{epoch} training loss = {avg_loss}, validation_loss = {val_loss}"
        )

        save_best = False

        if val_loss < best_loss:
            best_loss = val_loss
            save_best = True

        # Save intermediate checkpoint, so we can continue training if needed.
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
        }
        torch.save(checkpoint, CHECKPOINT_NAME)

        if save_best:
            print("Saving best loss model.")
            torch.save(checkpoint, BEST_CHECKPOINT_NAME)

        scheduler.step()
        # scheduler.step(avg_conflicts)

        # print(f"Current LR: {optimizer.param_groups[0]['lr']} Avg conflicts: {avg_conflicts}")
