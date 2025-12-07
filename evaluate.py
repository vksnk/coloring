from dataset import RigSetDataset
from loss import potts_loss, entropy_loss
from model import GCCN

import os

import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter

import networkx as nx
from torch_geometric.utils import to_networkx

BEST_CHECKPOINT_NAME = "checkpoints/best_checkpoint.pth"


def evaluate_networkx(batch, strategy):
    """
    Splits a PyG Batch, colors each graph individually to ensure
    local context, and re-assembles the results into a single Tensor.

    Args:
        batch (torch_geometric.data.Batch): Input batch.

    Returns:
        torch.Tensor: LongTensor of colors shape [total_num_nodes],
                      on the same device as batch.x.
    """
    # 1. Split the batch into a list of individual Data objects
    #    This automatically resets edge indices to start at 0 for each graph.
    data_list = batch.to_data_list()

    all_colors = []

    # 2. Iterate through each graph individually
    for data in data_list:
        # Convert to NetworkX (Move to CPU is automatic here)
        G = to_networkx(data, to_undirected=True)

        # Run coloring on this specific graph
        coloring_dict = nx.coloring.greedy_color(G, strategy=strategy)

        # Sort colors by node index (0 to num_nodes_in_this_graph)
        # and append to the master list
        graph_colors = [min(coloring_dict[i], 7) for i in range(data.num_nodes)]
        all_colors.extend(graph_colors)

    # 3. Convert the combined list back to a Tensor on the correct device
    return torch.tensor(all_colors, dtype=torch.long), 0.0


def wrap_evaluate_networkx(strategy):
    return lambda batch: evaluate_networkx(batch, strategy)


def evaluate_gnn(model, batch, device):
    # Process batch, apply softmax and find the most probable color assignment.
    batch = batch.to(device)
    logits = model(batch)
    loss = potts_loss(logits, batch.edge_index) + 0.02 * entropy_loss(logits)

    out = F.softmax(logits, dim=1)
    pred_colors = out.argmax(dim=1)

    return pred_colors, loss.item()


def wrap_evaluate_gnn(model, device):
    return lambda batch: evaluate_gnn(model, batch, device)


def evaluate_dataset(eval_f, loader):
    total_loss = 0.0

    total_conflicts = 0
    total_perfect_graphs = 0
    total_graphs = 0
    total_unsolvable = 0
    with torch.no_grad():
        for batch in loader:
            pred_color, loss = eval_f(batch)
            total_loss += loss

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

    return total_perfect_graphs, total_loss / len(loader)


if __name__ == "__main__":
    device = torch.device("cpu")
    pin_memory = False

    dataset = RigSetDataset("data/")
    test_dataset = dataset[dataset.test_mask]

    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=pin_memory
    )

    model = GCCN(dataset.num_classes).to(device)
    if os.path.exists(BEST_CHECKPOINT_NAME):
        checkpoint = torch.load(
            BEST_CHECKPOINT_NAME, weights_only=True, map_location=device
        )

        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best saved checkpoint.")
    else:
        assert False

    model.eval()
    num_correct, val_loss = evaluate_dataset(
        wrap_evaluate_gnn(model, device), test_loader
    )

    num_correct, val_loss = evaluate_dataset(
        wrap_evaluate_networkx("largest_first"), test_loader
    )

    num_correct, val_loss = evaluate_dataset(
        wrap_evaluate_networkx("random_sequential"), test_loader
    )

    num_correct, val_loss = evaluate_dataset(
        wrap_evaluate_networkx("smallest_last"), test_loader
    )

    num_correct, val_loss = evaluate_dataset(
        wrap_evaluate_networkx("independent_set"), test_loader
    )

    num_correct, val_loss = evaluate_dataset(
        wrap_evaluate_networkx("connected_sequential_bfs"), test_loader
    )

    num_correct, val_loss = evaluate_dataset(
        wrap_evaluate_networkx("connected_sequential_dfs"), test_loader
    )

    num_correct, val_loss = evaluate_dataset(
        wrap_evaluate_networkx("saturation_largest_first"), test_loader
    )
