from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

IMGS = Path(__file__).resolve().parents[1] / "data" / "processed" / "images"


MEDIA = [0.485, 0.456, 0.406]
DESVIO = [0.229, 0.224, 0.225]


def get_transforms(entrenamiento, tamano=224):
    if entrenamiento:
        return transforms.Compose([
            transforms.Resize((tamano, tamano)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEDIA, DESVIO),
        ])
    return transforms.Compose([
        transforms.Resize((tamano, tamano)),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA, DESVIO),
    ])


class LesionDataset(Dataset):
    def __init__(self, df, transform=None):
        self.ids = df["image_id"].tolist()
        self.etiquetas = df["etiqueta"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = Image.open(IMGS / f"{self.ids[i]}.jpg").convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.etiquetas[i], dtype=torch.float32)