import torch
import torch.nn as nn

class Verifier(nn.Module):
    def __init__(self, emb_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )

    def forward(self, img_emb, txt_emb):
        x = torch.cat([img_emb, txt_emb], dim=1)
        return self.net(x)