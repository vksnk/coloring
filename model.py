import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv, GINConv


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
