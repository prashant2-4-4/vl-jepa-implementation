#Text encoder for VL-JEPA Model


from sentence_transformers import SentenceTransformer
import torch

class TextEncoder:
    def __init__(self , device = 'cpu'):
        self.device = device
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.model.eval()
    
    @torch.no_grad()
    def encode(self, texts):
        emb = self.model.encode(texts , convert_to_tensor = True , normalize_embeddings = True)
        return emb