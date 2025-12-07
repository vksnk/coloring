import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, SAGEConv, GINConv


class GCCN(torch.nn.Module):
    def __init__(self, num_of_gcns, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.linear_input = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.num_conv_layers = num_of_gcns
        self.convs = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        for i in range(self.num_conv_layers):
            self.convs.append(
                SAGEConv(self.hidden_dim, self.hidden_dim, root_weight=True)
            )
            self.layer_norms.append(torch.nn.LayerNorm(self.hidden_dim))
        self.linear_output = torch.nn.Linear(
            (self.num_conv_layers + 1) * self.hidden_dim, num_classes
        )
        # self.linear_output = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        num_nodes, edge_index, batch = data.num_nodes, data.edge_index, data.batch

        x = torch.randn(num_nodes, self.input_dim).to(self.linear_input.weight.device)
        x = self.linear_input(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        xs = [x]
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
