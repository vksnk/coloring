import torch
import torch.nn.functional as F

from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, SAGEConv, GINConv


class GCCN(torch.nn.Module):
    """
    A Graph Convolutional Network that utilizes SAGEConv layers with residual
    connections where features from all layers are concatenated for the
    final classification.
    """

    def __init__(self, num_of_gcns, input_dim, hidden_dim, num_classes):
        """
        Initializes the architecture of the model.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # Project initial (random) features to the hidden dimension.
        self.linear_input = torch.nn.Linear(self.input_dim, self.hidden_dim)

        self.num_conv_layers = num_of_gcns
        self.convs = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()

        # Initialize convolution layers layers and corresponding LayerNorms.
        for i in range(self.num_conv_layers):
            self.convs.append(
                SAGEConv(self.hidden_dim, self.hidden_dim, root_weight=True)
            )
            self.layer_norms.append(torch.nn.LayerNorm(self.hidden_dim))

        # The output layer size is (N + 1) * hidden_dim because we concatenate
        # the initial embedding and the outputs of all N convolution layers.
        self.linear_output = torch.nn.Linear(
            (self.num_conv_layers + 1) * self.hidden_dim, num_classes
        )

    def forward(self, data):
        """
        Performs the forward pass by generating random initial node features,
        passing them through the graph convolutions, and concatenating multi-scale
        features for the final prediction.
        """
        num_nodes, edge_index, batch = data.num_nodes, data.edge_index, data.batch

        # Initialize node features with random noise.
        x = torch.randn(num_nodes, self.input_dim).to(self.linear_input.weight.device)

        # Initial projection and regularization.
        x = self.linear_input(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # List to store features from every layer (including the input projection).
        xs = [x]

        for i in range(self.num_conv_layers):
            # Store current state for the residual connection.
            x_in = x

            # Apply convolution layer.
            x = self.convs[i](x, edge_index)

            x = self.layer_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)

            # Add residual connection.
            x = x + x_in

            # Store the result of this layer to be used later.
            xs.append(x)

        # Concatenate features from all depth levels.
        x = torch.concat(xs, dim=1)

        # Project into final output vector.
        x = self.linear_output(x)

        return x


class GCCNWraper(torch.nn.Module):
    """
    This is just a helper object which wrap the model above to support the interface
    needed for model visualization library.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, batch_index):
        # Manually instantiate a Batch object.
        batch_obj = Batch(batch=batch_index, x=x, edge_index=edge_index)

        # Call wrapped model.
        return self.model(batch_obj)
