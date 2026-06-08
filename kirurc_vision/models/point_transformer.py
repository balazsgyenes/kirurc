from typing import List

import torch
from torch_cluster import knn_graph
from torch_geometric.data import Data
from torch_geometric.nn import MLP

from .modules.transformer_modules import TransformerBlock, TransitionDown, TransitionUp


class PointTransformer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dim_model: List[int],
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
        self.transformers_up = torch.nn.ModuleList()
        self.transformers_down = torch.nn.ModuleList()
        self.transition_up = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()

        for i in range(0, len(dim_model) - 1):

            # Add Transition Down block followed by a Point Transformer block
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

            # Add Transition Up block followed by Point Transformer block
            self.transition_up.append(
                TransitionUp(
                    in_channels=dim_model[i + 1],
                    out_channels=dim_model[i],
                )
            )

            self.transformers_up.append(
                TransformerBlock(
                    in_channels=dim_model[i],
                    out_channels=dim_model[i],
                )
            )

        # summit layers
        self.mlp_summit = MLP(
            [dim_model[-1], dim_model[-1]],
            norm=None,
            plain_last=False,
        )

        self.transformer_summit = TransformerBlock(
            in_channels=dim_model[-1],
            out_channels=dim_model[-1],
        )

        # class score computation
        self.mlp_output = MLP([dim_model[0], 64, out_channels], norm=None)

    def forward(self, data: Data):
        x, pos, batch = data.x, data.pos, data.batch
        # add dummy features in case there is none
        if x is None:
            x = torch.ones(pos.shape[0], 1, device=pos.device)

        out_x = []
        out_pos = []
        out_batch = []

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # save outputs for skipping connections
        out_x.append(x)
        out_pos.append(pos)
        out_batch.append(batch)

        # backbone down : #reduce cardinality and augment dimensionnality
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)
            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

            out_x.append(x)
            out_pos.append(pos)
            out_batch.append(batch)

        # summit
        x = self.mlp_summit(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_summit(x, pos, edge_index)

        # backbone up : augment cardinality and reduce dimensionnality
        n = len(self.transformers_down)
        for i in range(n):
            x = self.transition_up[-i - 1](
                x=out_x[-i - 2],
                x_sub=x,
                pos=out_pos[-i - 2],
                pos_sub=out_pos[-i - 1],
                batch_sub=out_batch[-i - 1],
                batch=out_batch[-i - 2],
            )

            edge_index = knn_graph(
                out_pos[-i - 2],
                k=self.k,
                batch=out_batch[-i - 2],
            )
            x = self.transformers_up[-i - 1](x, out_pos[-i - 2], edge_index)

        # Class score
        out = self.mlp_output(x)

        return out

    def freeze_mlp_head(self, freeze: bool, reset_weights: bool = False) -> None:
        self.mlp_output.requires_grad_(not freeze)
        if reset_weights:
            self.mlp_output.reset_parameters()

    def freeze_last_n_upsample_layers(
        self,
        n: int,
        freeze: bool,
        reset_weights: bool = False,
    ) -> None:
        for i in range(n):
            # e.g. transition_up[0] is the final layer in forward pass
            self.transition_up[i].requires_grad_(not freeze)
            self.transformers_up[i].requires_grad_(not freeze)
            if reset_weights:
                self.transition_up[i].mlp_sub.reset_parameters()
                self.transition_up[i].mlp.reset_parameters()
                self.transformers_up[i].lin_in.reset_parameters()
                self.transformers_up[i].lin_out.reset_parameters()
                self.transformers_up[i].pos_nn.reset_parameters()
                self.transformers_up[i].attn_nn.reset_parameters()
                self.transformers_up[i].transformer.reset_parameters()
