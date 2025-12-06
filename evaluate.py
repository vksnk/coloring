from dataset import RigSetDataset
from loss import potts_loss, entropy_loss
from model import GCCN

import os

import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

BEST_CHECKPOINT_NAME = "checkpoints/best_checkpoint.pth"


def evaluate_gnn(model, batch, device):
    # Process batch, apply softmax and find the most probable color assignment.
    batch = batch.to(device)
    logits = model(batch)
    loss = potts_loss(logits, batch.edge_index) + 0.02 * entropy_loss(logits)

    out = F.softmax(logits, dim=1)
    pred_colors = out.argmax(dim=1)

    return pred_colors, loss


def evaluate_dataset(eval_f, model, loader, device):
    model.eval()

    total_loss = 0.0

    total_conflicts = 0
    total_perfect_graphs = 0
    total_graphs = 0
    total_unsolvable = 0
    with torch.no_grad():
        for batch in loader:
            pred_color, loss = eval_f(model, batch, device)
            total_loss += loss.item()

            u, v = batch.edge_index
            node_counts = batch.batch.bincount()

            # Find conflicts for all edges at once.
            conflicts = pred_color[u] == pred_color[v]

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

    return total_perfect_graphs, total_loss / len(loader)


if __name__ == "__main__":
    device = torch.device("cpu")
    pin_memory = False

    dataset = RigSetDataset("data/")
    test_dataset = dataset[dataset.test_mask]

    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=pin_memory
    )

    model = GCCN(dataset.num_node_features, dataset.num_classes).to(device)
    if os.path.exists(BEST_CHECKPOINT_NAME):
        checkpoint = torch.load(
            BEST_CHECKPOINT_NAME, weights_only=True, map_location=device
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best saved checkpoint.")
    else:
        assert False

    num_correct, val_loss = evaluate_dataset(evaluate_gnn, model, test_loader, device)
