import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import os

class VisionTextDataset(Dataset):
    def __init__(self , csv_path , vision_preprocess):
        self.data = pd.read_csv(csv_path)
        self.process = vision_preprocess

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self , idx):
        row = self.data.iloc[idx]
        base_path = "my_image_archive/"
        image_path = base_path + row['image_name']
        caption = row['comment']

        image = Image.open(image_path).convert('RGB')
        image = self.process(image)

        return image , caption