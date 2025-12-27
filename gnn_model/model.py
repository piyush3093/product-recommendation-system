import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GraphConv, to_hetero


class GNN(nn.Module):
    def __init__(self, hidden_channels):

        super().__init__()
        self.conv1 = GraphConv(hidden_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:

        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = self.conv3(x, edge_index)

        return x


class Model(nn.Module):
    def __init__(self, hidden_dim, user_input_dim, item_input_dim, metadata):
        super().__init__()

        self.user_lin = nn.Linear(user_input_dim, hidden_dim)
        self.item_lin = nn.Linear(item_input_dim, hidden_dim)

        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_dim)
        self.gnn = to_hetero(self.gnn, metadata=metadata)
        

    def forward(self, data) -> torch.Tensor:

        x_dict = {
          "user": self.user_lin(data["user"].x),
          "item": self.item_lin(data["item"].x),
        }
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict, data.edge_weight_dict)

        return x_dict


def classifier(user_matrix, item_matrix, edge_matrix, edge_labels):

    classifier_out = (user_matrix[edge_matrix[0]] * item_matrix[edge_matrix[1]]).sum(dim=-1)

    return F.binary_cross_entropy_with_logits(classifier_out, edge_labels)