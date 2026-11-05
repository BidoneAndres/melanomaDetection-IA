from pathlib import Path
import torch
from PIL import Image, ImageDraw

from src.dataset import get_transforms
from src.modelo import crear_modelo

RAIZ = Path(__file__).resolve().parents[1]
MODELO_PATH = RAIZ / "models" / "cnn_resnet18.pt"
UMBRAL = 0.5

_modelo = None


def cargar_modelo(ruta=MODELO_PATH):
    global _modelo
    if _modelo is None:
        modelo = crear_modelo(preentrenado=False)
        modelo.load_state_dict(torch.load(ruta, map_location="cpu", weights_only=True))
        modelo.eval()
        _modelo = modelo
    return _modelo


@torch.no_grad()
def predecir(ruta_imagen, umbral=UMBRAL):
    ruta = Path(ruta_imagen)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe la imagen: {ruta}")

    img = Image.open(ruta).convert("RGB")
    x = get_transforms(entrenamiento=False)(img).unsqueeze(0)
    prob = torch.sigmoid(cargar_modelo()(x).squeeze()).item()

    return {
        "archivo": ruta.name,
        "probabilidad_melanoma": round(prob, 4),
        "etiqueta": "melanoma" if prob >= umbral else "nevo",
        "umbral": umbral,
    }


def guardar_imagen_etiquetada(ruta_imagen, resultado, salida):
    img = Image.open(ruta_imagen).convert("RGB")
    dibujo = ImageDraw.Draw(img)
    texto = f"{resultado['etiqueta']} ({resultado['probabilidad_melanoma']:.0%})"
    color = (255, 90, 90) if resultado["etiqueta"] == "melanoma" else (120, 220, 120)
    dibujo.rectangle([0, 0, img.width, 30], fill=(0, 0, 0))
    dibujo.text((8, 9), texto, fill=color)
    img.save(salida)