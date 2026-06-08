import torch
from torch_geometric.nn import MLP

from .modules.pointnet_modules import FPModule, GlobalSAModule, SAModule


class PointNetSeg(torch.nn.Module):
    def __init__(self, input_features: int, num_classes: int):
        super().__init__()
        # Input channels account for both `pos` and node features.
        self.sa1_module = SAModule(
            ratio=0.2,
            radius=0.2,
            nn=MLP([3 + input_features, 64, 64, 128]),
        )
        self.sa2_module = SAModule(
            ratio=0.25,
            radius=0.4,
            nn=MLP([128 + 3, 128, 128, 256]),
        )
        self.sa3_module = GlobalSAModule(nn=MLP([256 + 3, 256, 512, 1024]))

        self.fp3_module = FPModule(k=1, nn=MLP([1024 + 256, 256, 256]))
        self.fp2_module = FPModule(k=3, nn=MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(k=3, nn=MLP([128, 128, 128, 128]))

        self.mlp = MLP([128, 128, 128, num_classes], dropout=0.5, norm=None)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)

        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        return self.mlp(x)

    def freeze_mlp_head(self, freeze: bool, reset_weights: bool = False) -> None:
        self.mlp.requires_grad_(not freeze)
        if reset_weights:
            self.mlp.reset_parameters()

    def freeze_last_n_upsample_layers(self, 
        n: int,
        freeze: bool,
        reset_weights: bool = False,
    ) -> None:

        for i in range(n):
            # e.g. fp1_module is the final layer in forward pass
            fp_module: FPModule = getattr(self, f"fp{i+1}_module")
            fp_module.requires_grad_(not freeze)
            
            if reset_weights:
                fp_module.nn.reset_parameters()
