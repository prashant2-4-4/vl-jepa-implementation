import torch
import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self , vision_dim= 512 , text_dim = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim , 1024),
            nn.GELU(),
            nn.Linear(1024 , text_dim)
        )

    def forward(self , vision_emb):
        pred_text_emb = self.net(vision_emb)
        pred_text_emb = pred_text_emb / pred_text_emb.norm(dim = -1 , keepdim = True)
        return pred_text_emb
