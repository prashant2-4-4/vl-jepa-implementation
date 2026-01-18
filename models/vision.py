# Vision encoder for VL-JEPA Model
import torch
import clip

class VisionEncoder:
    def __init__(self , device = 'cpu'):
        self.device = device
        self.model , self.preprocess = clip.load("ViT-B/16", device=self.device)
        self.model.eval()
    
    @torch.no_grad()
    def encode(self, images):
        images = images.to(self.device)
        emb = self.model.encode_image(images)
        emb = emb / emb.norm(dim = -1 , keepdim = True)
        return emb