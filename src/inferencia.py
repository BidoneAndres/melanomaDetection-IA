import threading
from pathlib import Path

import torch
from PIL import Image, ImageDraw, UnidentifiedImageError

from src.dataset import get_transforms
from src.explicacion import mapa_calor_png_base64
from src.modelo import crear_modelo

RAIZ = Path(__file__).resolve().parents[1]
MODELO_PATH = RAIZ / "models" / "cnn_resnet18.pt"
UMBRAL = 0.7
AVISO = "Herramienta de apoyo; no reemplaza el diagnóstico médico."

_modelo = None
# El modelo se comparte entre pedidos el candado evita que dos hilos usen
# a la vez los hooks del Grad-CAM.
_candado = threading.Lock()


def cargar_modelo(ruta=MODELO_PATH):
    """Carga el modelo una sola vez y lo deja en modo evaluación."""
    global _modelo
    if _modelo is None:
        modelo = crear_modelo(preentrenado=False)
        modelo.load_state_dict(torch.load(ruta, map_location="cpu", weights_only=True))
        modelo.eval()
        _modelo = modelo
    return _modelo


def modelo_disponible():
    try:
        cargar_modelo()
        return True
    except FileNotFoundError:
        return False


def predecir_imagen(img, umbral=UMBRAL):
    """Recibe una imagen PIL y devuelve (probabilidad, etiqueta)."""
    x = get_transforms(entrenamiento=False)(img.convert("RGB")).unsqueeze(0)
    with _candado, torch.no_grad():
        prob = torch.sigmoid(cargar_modelo()(x).squeeze()).item()
    return prob, ("melanoma" if prob >= umbral else "nevo")


def explicar_imagen(img):
    """Mapa de calor Grad-CAM (PNG en base64)."""
    with _candado:
        return mapa_calor_png_base64(cargar_modelo(), img)


def predecir(ruta_imagen, umbral=UMBRAL):
    """Versión por archivo, usada por el script de consola."""
    ruta = Path(ruta_imagen)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe la imagen: {ruta}")
    try:
        with Image.open(ruta) as imagen:
            imagen.load()
            img = imagen.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError(f"El archivo no es una imagen válida: {ruta}") from error

    prob, etiqueta = predecir_imagen(img, umbral)
    return {
        "archivo": ruta.name,
        "probabilidad_melanoma": round(prob, 4),
        "etiqueta": etiqueta,
        "umbral": umbral,
    }


def guardar_imagen_etiquetada(ruta_imagen, resultado, salida):
    """Guarda la imagen con la etiqueta y la probabilidad en una barra superior."""
    img = Image.open(ruta_imagen).convert("RGB")
    dibujo = ImageDraw.Draw(img)
    texto = f"{resultado['etiqueta']} ({resultado['probabilidad_melanoma']:.0%})"
    color = (255, 90, 90) if resultado["etiqueta"] == "melanoma" else (120, 220, 120)
    dibujo.rectangle([0, 0, img.width, 30], fill=(0, 0, 0))
    dibujo.text((8, 9), texto, fill=color)
    img.save(salida)