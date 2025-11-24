from dataset import RigSetDataset

import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GCCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # print(x.shape, edge_index)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.softmax(x, dim=1)


def potts_loss(h, edge_index):
    u = edge_index[0]
    v = edge_index[1]
    h_u = h[u]
    h_v = h[v]

    # print(h_u[0:10])
    prod = h_u * h_v
    prod = prod.sum(dim=1)
    # print(h_u.shape, h_v.shape, prod.shape)
    return torch.mean(prod)


if __name__ == "__main__":
    # if torch.mps.is_available():
    #     print('Using mps backend...')
    #     device = torch.device("mps")
    # elif torch.cuda.is_available():
    #     print('Using cuda backend...')
    #     device = torch.device("cuda")
    # else:
    print("Using cpu backend...")
    device = torch.device("cpu")

    dataset = RigSetDataset("data/")
    # dataset = Planetoid(root='/tmp/Cora', name='Cora')
    loader = DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )

    model = GCCN().to(device)
    # data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()

    for epoch in range(200):
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = potts_loss(out, batch.edge_index)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch #{epoch} loss: {avg_loss}")

    # model.eval()
    # pred = model(data).argmax(dim=1)
    # correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    # acc = int(correct) / int(data.test_mask.sum())
    # print(f'Accuracy: {acc:.4f}')
