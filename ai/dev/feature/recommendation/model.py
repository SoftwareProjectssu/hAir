#model.py

import torch
import torch.nn as nn 

class FeatureEncode(nn.Module):
    def __init__(self, fs_dim=6, fr_dim=8, pref_dim=5, emb_dim=16):
        super().__init__()
        self.fs_emb = nn.Embedding(fs_dim, emb_dim)
        self.totalDim = emb_dim + fr_dim + pref_dim

    def forward(self, fsIdx, fr, prefs):
        fs_vec = self.fs_emb(fsIdx)
        return torch.cat([fs_vec, fr, prefs], dim=1)
    
class RankingML(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], 1)
        )

    def forward(self, x):
        return self.seq(x)

