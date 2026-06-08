import torch
from torch_cluster import knn_graph
from torch_geometric.data import Data
from torch_geometric.nn import (
    MLP,
    global_mean_pool,
)

from .modules.transformer_modules import TransformerBlock, TransitionDown


class RegressionPointTransformer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_targets: int,
        dim_model: list[int],
        k: int = 16,
    ):
        super().__init__()
        self.k = k

        # dummy feature is created if there is none given
        in_channels = max(in_channels, 1)

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]], plain_last=False)

        self.transformer_input = TransformerBlock(
            in_channels=dim_model[0],
            out_channels=dim_model[0],
        )
        # backbone layers
        self.transformers_down = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()

        for i in range(len(dim_model) - 1):
            # Add Transition Down block followed by a Transformer block
            self.transition_down.append(
                TransitionDown(
                    in_channels=dim_model[i],
                    out_channels=dim_model[i + 1],
                    k=self.k,
                )
            )

            self.transformers_down.append(
                TransformerBlock(
                    in_channels=dim_model[i + 1],
                    out_channels=dim_model[i + 1],
                )
            )

        # class score computation
        self.mlp_output = MLP([dim_model[-1], 64, out_targets * 3], norm=None)

    def forward(self, data: Data):
        x, pos, batch = data.x, data.pos, data.batch
        # add dummy features in case there is none
        if x is None:
            x = torch.ones((pos.shape[0], 1), device=pos.get_device())

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # backbone
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)
            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

        # GlobalAveragePooling
        x = global_mean_pool(x, batch)

        # Class score
        out = self.mlp_output(x)

        B, *_ = out.shape
        return out.view(B, -1, 3)

    def freeze_mlp_head(self, freeze: bool, reset_weights: bool = False) -> None:
        self.mlp_output.requires_grad_(not freeze)
        if reset_weights:
            self.mlp_output.reset_parameters()
