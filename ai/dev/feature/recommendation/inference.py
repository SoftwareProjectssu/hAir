#inference.py

import torch
from model import FeatureEncode, RankingML

def predict(encode, recommender, fs, fr, prefs, styles, k=3):
    with torch.no_grad():
        usr_feat = encode(fs, fr, prefs)
        input_tensor = torch.cat([usr_feat.repeat(len(styles), 1), styles], dim=1)
        scores = recommender(input_tensor).squeeze()
        top = torch.topk(scores, k=k)
        return top.indices, top.values
    
fs = torch.tensor([2])
fr = torch.randn(1, 8)
prefs = torch.randn(1, 5)
styles = torch.randn(10, 16)  # 10 styles to compare

encoder = FeatureEncode()
ranking_model = RankingML(encoder.totalDim + 16, [128, 64])

topIdx, topScores = predict(encoder, ranking_model, fs, fr, prefs, styles)
print("Top indices:", topIdx.tolist(), topScores.tolist())

