"""Small projection head trained on top of frozen DINOv2 features (triplet loss)."""
import torch
from torch import nn


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=1536, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, out_dim))

    def forward(self, x):
        return self.net(x)


def save_head(head: ProjectionHead, path, in_dim, out_dim):
    torch.save({"in_dim": in_dim, "out_dim": out_dim, "weights": head.state_dict()}, path)
