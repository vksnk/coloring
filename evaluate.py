import loss as l

import os

import torch
import torch.nn.functional as F

from torch_geometric.utils import scatter


def evaluate_dataset(model, loader, device):
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
            loss = l.potts_loss(logits, batch.edge_index) + 0.02 * l.entropy_loss(
                logits
            )
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

    return total_perfect_graphs, total_loss / len(loader)
