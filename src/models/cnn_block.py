import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, n_in, n_out, k, drop):
        super().__init__()
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(n_in, n_out, k, padding=p),
            nn.BatchNorm1d(n_out),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv1d(n_out, n_out, k, padding=p),
            nn.BatchNorm1d(n_out),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x.permute(0, 2, 1)).permute(0, 2, 1)