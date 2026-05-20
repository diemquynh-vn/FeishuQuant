import torch
import torch.nn as nn
from src.models.cnn_block import CNNBlock
from src.models.attention import EnhancedAttention

class CNNLSTMModel(nn.Module):
    def __init__(self, n_feat, hidden=64, n_layer=2, c_chan=32, k=3, drop=0.4):
        super().__init__()

        self.cnn = CNNBlock(n_feat, c_chan, k, drop)
        self.lstm = nn.LSTM(c_chan, hidden, n_layer, batch_first=True,
                            dropout=drop if n_layer > 1 else 0, bidirectional=False)
        self.attn = EnhancedAttention(hidden, num_heads=4)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 32),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(32, 1)
        )
        self._init()

    def _init(self):
        for n, p in self.named_parameters():
            if 'weight' in n and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif 'bias' in n:
                nn.init.zeros_(p)

    def forward(self, x):
        x = self.cnn(x)
        x, _ = self.lstm(x)
        x = self.attn(x)
        return self.head(x).squeeze(-1)