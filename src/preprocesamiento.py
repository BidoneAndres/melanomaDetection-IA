from pathlib import Path
import numpy as np
import cv2


IMGS = Path(__file__).resolve().parents[1] / "data" / "processed" / "images"


def cargar(image_id):
    img = cv2.imread(str(IMGS / f"{image_id}.jpg"))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def quitar_pelos(img, kernel_size=17, umbral=20):
    
    gris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gris, cv2.MORPH_BLACKHAT, kernel)
    _, mascara_pelos = cv2.threshold(blackhat, umbral, 255, cv2.THRESH_BINARY)
    mascara_pelos = cv2.dilate(mascara_pelos, np.ones((3, 3), np.uint8))
    limpia = cv2.inpaint(img, mascara_pelos, 3, cv2.INPAINT_TELEA)
    return limpia, mascara_pelos


def mascara_esquinas(img, umbral=25):
   
    gris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    oscuro = (gris < umbral).astype(np.uint8)
    n, etiquetas = cv2.connectedComponents(oscuro)
    borde = np.zeros_like(oscuro)
    h, w = oscuro.shape
    for (y, x) in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        if etiquetas[y, x] != 0:
            borde[etiquetas == etiquetas[y, x]] = 1
    borde = cv2.dilate(borde, np.ones((5, 5), np.uint8))
    return ((1 - borde) * 255).astype(np.uint8)


def segmentar_otsu(img, valida):
    
    gris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    fondo = cv2.GaussianBlur(gris.astype(np.float32), (0, 0), sigmaX=80)
    corr = gris.astype(np.float32) / (fondo + 1e-6)
    gris = cv2.normalize(corr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    suave = cv2.GaussianBlur(gris, (7, 7), 0)

    valores = suave[valida > 0].reshape(-1, 1)
    umbral, _ = cv2.threshold(valores, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mascara = ((suave < umbral) & (valida > 0)).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, k)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, k)

    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mascara)
    if n > 1:
        mayor = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mascara = ((etiquetas == mayor) * 255).astype(np.uint8)
    return mascara


def procesar(img):
    """Pipeline completo. Devuelve (limpia, m_pelos, valida, lesion)."""
    limpia, m_pelos = quitar_pelos(img)
    valida = mascara_esquinas(limpia)
    lesion = segmentar_otsu(limpia, valida)
    return limpia, m_pelos, valida, lesion