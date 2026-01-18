# eval.py
import torch
from PIL import Image

from models.text import TextEncoder
from models.vision import VisionEncoder
from models.predictor import Predictor


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize models
    vision = VisionEncoder(device=device)
    text = TextEncoder(device=device)
    predictor = Predictor().to(device)
    predictor.eval()

    # Load and preprocess image
    image_path = r"my_image_archive\10002456.jpg"
    image = Image.open(image_path).convert("RGB")
    image_tensor = vision.preprocess(image).unsqueeze(0).to(device)

    # Encode image
    with torch.no_grad():
        vision_emb = vision.encode(image_tensor)
        pred_emb = predictor(vision_emb)

    # Candidate labels
    labels = [
        "persons hanging bridge and cable",
        "person stirring pan",
        "person washing dishes"
    ]

    # Encode labels
    with torch.no_grad():
        label_embs = text.encode(labels)

    # Cosine similarity
    scores = torch.cosine_similarity(pred_emb, label_embs, dim=-1)

    print("Prediction:", labels[scores.argmax().item()])
    print("Scores:", scores.cpu().tolist())


if __name__ == "__main__":
    main()

