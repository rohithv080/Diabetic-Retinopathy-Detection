import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class DRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, apply_clahe=True, apply_cropping=True):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.apply_clahe = apply_clahe
        self.apply_cropping = apply_cropping

    def __len__(self):
        return len(self.df)

    def apply_clahe_rgb(self, image):
        # Convert RGB to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge back and convert to RGB
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    def __getitem__(self, idx):
        # Get image id and label
        img_id = self.df.iloc[idx]['id_code']
        label = self.df.iloc[idx]['diagnosis']
        
        # The paper uses binary classification: 0 (No DR) vs 1-4 (DR)
        binary_label = 0 if label == 0 else 1
        
        # Load image (assuming .png extension, might need to adjust if they are .jpeg)
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        if not os.path.exists(img_path):
             img_path = os.path.join(self.img_dir, f"{img_id}.jpeg")
                
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize all images to the same size (224x224 for Swin Transformer)
        image = cv2.resize(image, (224, 224))

        if self.apply_clahe:
            image = self.apply_clahe_rgb(image)
            
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # Convert HxWxC to CxHxW and scale to [0, 1] for PyTorch
            image = image.transpose((2, 0, 1))
            image = torch.from_numpy(image).float() / 255.0
            
        return image, torch.tensor(binary_label, dtype=torch.long)

def get_dataloaders(csv_path, img_dir, transform=None, batch_size=32, num_workers=2):
    df = pd.read_csv(csv_path)
    dataset = DRDataset(df, img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return dataloader
