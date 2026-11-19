import base64
import io

import numpy as np
import torch
from PIL import Image

from src.dataset import get_transforms


def mapa_gradcam(modelo, img):
    """Grad-CAM sobre el último bloque de la ResNet18.

    Devuelve una matriz con valores entre 0 y 1 (7x7). Los valores altos marcan
    las zonas que más empujaron la salida hacia 'melanoma'.
    """
    x = get_transforms(entrenamiento=False)(img.convert("RGB")).unsqueeze(0)
    activaciones, gradientes = [], []

    def hook_forward(modulo, entrada, salida):
        activaciones.append(salida)
        salida.register_hook(lambda g: gradientes.append(g))

    manejador = modelo.layer4.register_forward_hook(hook_forward)
    try:
        with torch.enable_grad():
            modelo.zero_grad()
            logit = modelo(x).squeeze()
            logit.backward()
    finally:
        manejador.remove()

    a = activaciones[0].detach()[0]
    g = gradientes[0].detach()[0]
    pesos = g.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((pesos * a).sum(dim=0))
    maximo = cam.max()
    if maximo > 0:
        cam = cam / maximo
    return cam.numpy()


def superponer_mapa(img, cam):
    """Pinta el mapa (amarillo a rojo) sobre la imagen original."""
    base = np.asarray(img.convert("RGB"), dtype=np.float32)
    mapa = Image.fromarray((cam * 255).astype(np.uint8)).resize(
        img.size, Image.Resampling.BILINEAR
    )
    m = np.asarray(mapa, dtype=np.float32) / 255.0
    calor = np.zeros_like(base)
    calor[..., 0] = 255
    calor[..., 1] = 255 * (1 - m)
    alfa = (0.65 * m)[..., None]
    mezcla = base * (1 - alfa) + calor * alfa
    return Image.fromarray(mezcla.astype(np.uint8))


def mapa_calor_png_base64(modelo, img):
    salida = superponer_mapa(img, mapa_gradcam(modelo, img))
    buf = io.BytesIO()
    salida.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")