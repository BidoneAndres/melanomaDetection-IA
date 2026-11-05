import numpy as np
import torch
from sklearn.metrics import recall_score, f1_score


def entrenar_epoca(modelo, loader, criterio, optimizador, device):
    modelo.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizador.zero_grad()
        loss = criterio(modelo(x).squeeze(1), y)
        loss.backward()
        optimizador.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluar(modelo, loader, criterio, device):
    modelo.eval()
    total, probs, reales = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = modelo(x).squeeze(1)
        total += criterio(logits, y).item() * len(y)
        probs.extend(torch.sigmoid(logits).cpu().numpy())
        reales.extend(y.cpu().numpy())
    probs, reales = np.array(probs), np.array(reales)
    pred = (probs >= 0.5).astype(int)
    return {
        "loss": total / len(loader.dataset),
        "recall": recall_score(reales, pred),
        "f1_macro": f1_score(reales, pred, average="macro"),
    }

@torch.no_grad()
def probabilidades(modelo, loader, device):
    """Devuelve las probabilidades de melanoma y las etiquetas reales."""
    modelo.eval()
    probs, reales = [], []
    for x, y in loader:
        logits = modelo(x.to(device)).squeeze(1)
        probs.extend(torch.sigmoid(logits).cpu().numpy())
        reales.extend(y.numpy())
    return np.array(probs), np.array(reales).astype(int)