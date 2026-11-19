import base64
import hashlib
import io
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src import inferencia
from src.schemas import Explicacion, Foto, RespuestaPrediccion, Resultado

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("melanoma.api")

MAX_BYTES = 10 * 1024 * 1024
FORMATOS = {"JPEG": "image/jpeg", "PNG": "image/png"}

NOTA_EXPLICACION = (
    "Zonas de la imagen que más aportaron a la probabilidad de melanoma según el "
    "modelo (Grad-CAM). Indica dónde miró la red; no es una explicación clínica "
    "y no fue validada con dermatólogos."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Modelo disponible: %s", inferencia.modelo_disponible())
    yield


app = FastAPI(
    title="Clasificación asistida de melanoma",
    version="0.3.0",
    description=(
        "Recibe una imagen dermatoscópica y un identificador de paciente, y devuelve "
        "la probabilidad de melanoma. Herramienta de apoyo; no reemplaza el "
        "diagnóstico médico. No se guarda ningún dato."
    ),
    lifespan=lifespan,
)


@app.get("/salud", summary="Estado del servicio")
def salud():
    return {"estado": "ok", "modelo_cargado": inferencia.modelo_disponible()}


@app.post(
    "/predicciones",
    response_model=RespuestaPrediccion,
    summary="Agregar una foto y obtener la predicción",
    responses={
        400: {"description": "El archivo está vacío o no es una imagen válida"},
        413: {"description": "La imagen supera los 10 MB"},
        415: {"description": "Formato no soportado (solo JPEG o PNG)"},
        422: {"description": "Datos del formulario inválidos"},
        503: {"description": "El modelo no está disponible"},
    },
)
def crear_prediccion(
    foto: Annotated[UploadFile, File(description="Imagen dermatoscópica (JPEG o PNG, hasta 10 MB)")],
    paciente: Annotated[
        str,
        Form(
            min_length=1,
            max_length=64,
            pattern=r"^\w[\w\-. ]*$",
            description="Identificador del paciente (usar un código, no datos personales)",
        ),
    ],
    explicar: Annotated[bool, Form(description="Incluir el mapa de calor (Grad-CAM)")] = False,
):
    contenido = foto.file.read(MAX_BYTES + 1)
    if len(contenido) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="La imagen supera los 10 MB.")
    if not contenido:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")

    try:
        img = Image.open(io.BytesIO(contenido))
        formato = img.format
        img.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="El archivo no es una imagen válida.")
    if formato not in FORMATOS:
        raise HTTPException(status_code=415, detail="Formato no soportado: usar JPEG o PNG.")
    img = img.convert("RGB")

    if not inferencia.modelo_disponible():
        raise HTTPException(status_code=503, detail="El modelo no está disponible.")

    prob, etiqueta = inferencia.predecir_imagen(img)
    explicacion = None
    if explicar:
        explicacion = Explicacion(
            metodo="Grad-CAM",
            mapa_calor_base64=inferencia.explicar_imagen(img),
            nota=NOTA_EXPLICACION,
        )

    sha = hashlib.sha256(contenido).hexdigest()
    # No se registra el identificador del paciente.
    logger.info("prediccion sha256=%s etiqueta=%s prob=%.3f explicar=%s", sha[:12], etiqueta, prob, explicar)

    return RespuestaPrediccion(
        paciente=paciente,
        resultado=Resultado(
            probabilidad_melanoma=round(prob, 4),
            etiqueta=etiqueta,
            umbral=inferencia.UMBRAL,
        ),
        foto=Foto(
            nombre_archivo=foto.filename or "sin_nombre",
            tipo_mime=FORMATOS[formato],
            ancho=img.width,
            alto=img.height,
            sha256=sha,
            contenido_base64=base64.b64encode(contenido).decode("ascii"),
        ),
        fecha_hora=datetime.now(timezone.utc),
        explicacion=explicacion,
        aviso=inferencia.AVISO,
    )