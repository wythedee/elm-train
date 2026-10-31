"""Inference-only native-channel encoders; no label/dataset input to forward."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from .channels import scatter_channels
from .vendor.cbramod import CBraMod


def backbone(seq_len=10):
    net = CBraMod(in_dim=200, out_dim=200, d_model=200,
                  dim_feedforward=800, seq_len=seq_len, n_layer=12, nhead=8)
    net.proj_out = nn.Identity()
    return net


class BottleneckResidualAdapter(nn.Module):
    def __init__(self, dim=50, bottleneck=24):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(self.norm(x))))


class FrozenLabelEncoder(nn.Module):
    """16 CBraMod experts, raw-stat Top3 router, 200→50, two anchors/channel."""
    def __init__(self, experts=16, router_patches=10, router_hidden=128,
                 router_top_k=3, adapter_dim=24, temperature=1.0):
        super().__init__()
        self.experts = nn.ModuleList(backbone(router_patches) for _ in range(experts))
        self.router = nn.Sequential(nn.Flatten(),
                                    nn.Linear(75 * router_patches * 2, router_hidden),
                                    nn.GELU(), nn.Dropout(0.1),
                                    nn.Linear(router_hidden, experts))
        self.token_compressor = nn.Linear(200, 50, bias=False)
        self.lowdim_adapters = nn.ModuleList(
            BottleneckResidualAdapter(50, adapter_dim) for _ in range(experts))
        self.router_patches, self.top_k = router_patches, router_top_k
        self.temperature = temperature
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x, indices):
        if x.ndim != 4 or x.shape[-1] != 200:
            raise ValueError("Expected [batch,native_channels,patches,200]")
        if self.training:
            raise RuntimeError("The released Encoder is frozen; call eval()")
        features = torch.stack((x.mean(-1), (x.square().mean(-1) + 1e-6).log()), -1)
        batch, channels, patches, _ = features.shape
        if patches != self.router_patches:
            temporal = features.permute(0, 1, 3, 2).reshape(batch, channels * 2, patches)
            temporal = F.interpolate(temporal, size=self.router_patches,
                                     mode="linear", align_corners=False)
            features = temporal.reshape(batch, channels, 2, self.router_patches).permute(0, 1, 3, 2)
        features, _ = scatter_channels(features, indices, 75)
        soft = torch.softmax(self.router(features) / self.temperature, -1)
        chosen = soft.topk(self.top_k, dim=-1).indices
        route = torch.zeros_like(soft).scatter(1, chosen, soft.gather(1, chosen))
        route = route / route.sum(-1, keepdim=True).clamp_min(1e-12)
        banks, valid = [], None
        for expert, adapter in zip(self.experts, self.lowdim_adapters):
            f = expert(x)
            anchors = torch.tensor([min(f.shape[2] - 1, int((i + 0.5) * f.shape[2] / 2))
                                    for i in range(2)], device=x.device)
            z = adapter(self.token_compressor(f.index_select(2, anchors)))
            z, valid = scatter_channels(z, indices, 75)
            banks.append(z)
        z = (torch.stack(banks, 1) * route[:, :, None, None, None]).sum(1)
        valid = valid[:, :, None].expand(-1, -1, 2).reshape(batch, 150)
        return z.reshape(batch, 1, 150, 50) * valid[:, None, :, None], valid, route


class FrozenStatisticsEncoder(nn.Module):
    def __init__(self, adapter_dim=24):
        super().__init__()
        self.backbone = backbone(4)
        self.compressor = nn.Linear(200, 50, bias=False)
        self.adapter = BottleneckResidualAdapter(50, adapter_dim)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x, indices, expected_valid=None):
        if self.training:
            raise RuntimeError("The released Encoder is frozen; call eval()")
        f = self.backbone(x)
        # This branch historically uses first/last anchors; do not replace it
        # with the class branch's two bin centers.
        anchors = torch.linspace(0, f.shape[2] - 1, 2, device=f.device).round().long()
        z = self.adapter(self.compressor(f.index_select(2, anchors)))
        z, valid = scatter_channels(z, indices, 75)
        valid = valid[:, :, None].expand(-1, -1, 2).reshape(len(x), 150)
        if expected_valid is not None:
            valid = valid & expected_valid
        return z.reshape(len(x), 1, 150, 50) * valid[:, None, :, None], valid


def load_encoders(class_file, statistics_file, config, device="cuda"):
    from safetensors.torch import load_file
    label = FrozenLabelEncoder(**config)
    stats = FrozenStatisticsEncoder(config.get("adapter_dim", 24))
    label.load_state_dict(load_file(str(class_file)), strict=True)
    stats.load_state_dict(load_file(str(statistics_file)), strict=True)
    return label.to(device).eval(), stats.to(device).eval()
