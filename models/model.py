import torch
import torch.nn.functional as F
import torch.nn as nn

# custom module
from models.layers import (
    TimeEncoder,
    SAGEConv,
)


def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    else:
        raise ValueError("Unknown activation")


class GEARSage(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        edge_attr_channels=50,
        time_channels=50,
        num_layers=2,
        dropout=0.0,
        bn=True,
        activation="elu",
    ):

        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        bn = nn.BatchNorm1d if bn else nn.Identity

        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            self.convs.append(
                SAGEConv(
                    (
                        first_channels + edge_attr_channels + time_channels,
                        first_channels,
                    ),
                    second_channels,
                )
            )
            self.bns.append(bn(second_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)
        self.emb_type = nn.Embedding(12, edge_attr_channels)
        self.emb_direction = nn.Embedding(2, edge_attr_channels)
        self.t_enc = TimeEncoder(time_channels)
        self.reset_parameters()

    def reset_parameters(self):

        for conv in self.convs:
            conv.reset_parameters()

        for bn in self.bns:
            if not isinstance(bn, nn.Identity):
                bn.reset_parameters()

        nn.init.xavier_uniform_(self.emb_type.weight)

        nn.init.xavier_uniform_(self.emb_direction.weight)

    def forward(self, x, edge_index, edge_attr, edge_t, edge_d):
        edge_attr = self.emb_type(edge_attr) + self.emb_direction(edge_d)
        edge_t = self.t_enc(edge_t)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr, edge_t)
            x = self.bns[i](x)
            x = self.activation(x)
            x = self.dropout(x)

        return x.log_softmax(dim=-1)
