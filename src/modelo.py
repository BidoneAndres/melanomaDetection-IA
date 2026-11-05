import torch.nn as nn
from torchvision import models


def crear_modelo(preentrenado=True):
    pesos = models.ResNet18_Weights.DEFAULT if preentrenado else None
    modelo = models.resnet18(weights=pesos)
    modelo.fc = nn.Linear(modelo.fc.in_features, 1)
    return modelo


def congelar_base(modelo):
    for p in modelo.parameters():
        p.requires_grad = False
    for p in modelo.fc.parameters():
        p.requires_grad = True


def descongelar_ultimo_bloque(modelo):
    for p in modelo.layer4.parameters():
        p.requires_grad = True