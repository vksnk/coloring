from dataset import RigSetDataset

import os

import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv


class GCCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        num_hidden_features = 64
        self.linear_input = torch.nn.Linear(num_node_features, num_hidden_features)
        self.conv1 = SAGEConv(
            num_hidden_features, num_hidden_features, root_weight=True
        )
        self.layer_norm1 = torch.nn.LayerNorm(num_hidden_features)
        self.conv2 = SAGEConv(
            num_hidden_features, num_hidden_features, root_weight=True
        )
        self.layer_norm2 = torch.nn.LayerNorm(num_hidden_features)
        self.conv3 = SAGEConv(
            num_hidden_features, num_hidden_features, root_weight=True
        )
        self.layer_norm3 = torch.nn.LayerNorm(num_hidden_features)
        self.linear_output = torch.nn.Linear(num_hidden_features, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.linear_input(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x_in = x
        x = self.conv1(x, edge_index)
        x = self.layer_norm1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = x + x_in

        x_in = x
        x = self.conv2(x, edge_index)
        x = self.layer_norm2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = x + x_in

        x_in = x
        x = self.conv3(x, edge_index)
        x = self.layer_norm3(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = x + x_in

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


CHECKPOINT_NAME = "checkpoints/checkpoint.pth"

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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

    if os.path.exists(CHECKPOINT_NAME):
        checkpoint = torch.load(CHECKPOINT_NAME, weights_only=True, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Loaded saved checkpoint from epoch #{start_epoch - 1}")
    else:
        start_epoch = 0

    model.train()

    for epoch in range(start_epoch, 50):
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
        print(f"Epoch #{epoch} loss: {avg_loss}")

        # Save intermediate checkpoint, so we can continue training if needed.
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }
        torch.save(checkpoint, CHECKPOINT_NAME)

        # torch.set_printoptions(profile="full")
        # print(F.softmax(out, dim=1)[0:25])
        # torch.set_printoptions(profile="default")

    model.eval()
    # model.train()
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            print(batch.x)
            logits = model(batch)
            print("Max Logit:", logits.max().item())
            print("Min Logit:", logits.min().item())
            out = F.softmax(logits, dim=1)
            hard_colors = out.argmax(dim=1)
            print(out)
            print(hard_colors)

    # pred = model(data).argmax(dim=1)
    # correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    # acc = int(correct) / int(data.test_mask.sum())
    # print(f'Accuracy: {acc:.4f}')
