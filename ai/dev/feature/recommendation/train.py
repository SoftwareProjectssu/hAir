#train.py

import torch
import torch.nn as nn
import torch.optim as opt
from model import FeatureEncode, RankingML

fs_dim, fr_dim, pref_dim = 6, 8, 5
hidden_dim = [128, 64]

encoder = FeatureEncode(fs_dim, fr_dim, pref_dim)
ranking_model = RankingML(encoder.totalDim + 16, hidden_dim) # style embedding 16 added

# dummy data
fs = torch.randint(0, fs_dim, (32,))
fr = torch.randn(32, fr_dim)
prefs = torch.randn(32, pref_dim)
style = torch.randn(32, 16)
labels = torch.randn(32, 1) # dummy labels

criterion = nn.BCEWithLogitsLoss()
optimizer = opt.Adam(list(encoder.parameters()) + list(ranking_model.parameters()), lr=1e-3)

for epoch in range(10):
    usr_feat = encoder(fs, fr, prefs)
    input_tensor = torch.cat([usr_feat, style], dim=1)
    out = ranking_model(input_tensor)
    loss = criterion(out, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')



