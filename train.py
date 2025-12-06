from dataset import RigSetDataset

import os

import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, GINConv
from torch_geometric.utils import scatter


class GCCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        hidden_dim = 128
        self.linear_input = torch.nn.Linear(num_node_features, hidden_dim)
        self.num_conv_layers = 3
        self.convs = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        for i in range(self.num_conv_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, root_weight=True))
            self.layer_norms.append(torch.nn.LayerNorm(hidden_dim))
        self.linear_output = torch.nn.Linear(
            (self.num_conv_layers + 0) * hidden_dim, num_classes
        )
        # self.linear_output = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.linear_input(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        xs = []
        for i in range(self.num_conv_layers):
            x_in = x
            x = self.convs[i](x, edge_index)
            x = self.layer_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = x + x_in
            xs.append(x)

        x = torch.concat(xs, dim=1)
        x = self.linear_output(x)

        return x


def potts_loss(h, edge_index):
    h = F.softmax(h, dim=1)
    # print(h[0:10])
    u = edge_index[0]
    v = edge_index[1]
    h_u = h[u]
    h_v = h[v]

    prod = h_u * h_v
    prod = prod.sum(dim=1)

    return torch.mean(prod)


def entropy_loss(h):
    p = F.softmax(h, dim=1)
    log_p = F.log_softmax(h, dim=1)

    entropy = -1 * (p * log_p).sum(dim=1)
    return torch.mean(entropy)


def evaluate_dataset(model, loader):
    model.eval()

    total_loss = 0.0

    total_conflicts = 0
    total_perfect_graphs = 0
    total_graphs = 0
    total_unsolvable = 0
    with torch.no_grad():
        for batch in loader:
            # Process batch, apply softmax and find the most probable color assignment.
            batch = batch.to(device)
            logits = model(batch)
            loss = potts_loss(logits, batch.edge_index) + 0.02 * entropy_loss(logits)
            total_loss += loss.item()

            out = F.softmax(logits, dim=1)
            hard_colors = out.argmax(dim=1)

            u, v = batch.edge_index
            node_counts = batch.batch.bincount()

            # Find conflicts for all edges at once.
            conflicts = hard_colors[u] == hard_colors[v]

            total_conflicts += conflicts.sum().item()

            # Aggregate mistakes per graph.
            edge_batch = batch.batch[batch.edge_index[0]]
            mistakes_per_graph = scatter(
                conflicts.long(), edge_batch, dim=0, reduce="sum"
            )

            # Count graphs with 0 mistakes.
            perfect_graphs_in_batch = (mistakes_per_graph == 0).sum().item()

            # Update totals.
            total_perfect_graphs += perfect_graphs_in_batch
            total_graphs += batch.num_graphs
            total_unsolvable += (batch.yk > 8).sum().item()
            # print(batch.yk)

            # print(f"Batch: {perfect_graphs_in_batch}/{batch.num_graphs} perfect.")

        print(
            f"Total: {total_perfect_graphs}/{total_graphs} perfect with {total_unsolvable} unsolvable."
        )

    model.train()

    return total_perfect_graphs, total_loss / len(train_loader)


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

        num_correct, val_loss = evaluate_dataset(model, val_loader)

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
