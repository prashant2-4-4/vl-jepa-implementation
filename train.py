import torch
from torch.optim import AdamW
from models.vision import VisionEncoder
from models.text import TextEncoder
from models.predictor import Predictor
from torch.utils.data import DataLoader
from models.dataset import VisionTextDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vision = VisionEncoder(device=device)
text = TextEncoder(device=device)
predictor = Predictor().to(device)

optimizer = AdamW(predictor.parameters() , lr = 1e-4)

dataset = VisionTextDataset(csv_path="unique_caption.csv" , vision_preprocess= vision.preprocess)

dataloader = DataLoader(dataset , batch_size=16 , shuffle=True)


for epoch in range(10):
    for images, caption in dataloader:
        images = images.to(device)
        with torch.no_grad():
            vision_emb = vision.encode(images)
            text_emb = text.encode(caption).to(device)

        pred = predictor(vision_emb)

        loss = 1 - torch.cosine_similarity(pred , text_emb , dim = -1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} , Loss: {loss.item():.4f}")