from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class Resultado(BaseModel):
    probabilidad_melanoma: float = Field(ge=0, le=1)
    etiqueta: str = Field(description="'melanoma' o 'nevo'")
    umbral: float


class Foto(BaseModel):
    nombre_archivo: str
    tipo_mime: str
    ancho: int
    alto: int
    sha256: str
    contenido_base64: str


class Explicacion(BaseModel):
    metodo: str
    mapa_calor_base64: str = Field(description="PNG codificado en base64")
    nota: str


class RespuestaPrediccion(BaseModel):
    paciente: str
    resultado: Resultado
    foto: Foto
    fecha_hora: datetime = Field(description="Momento de la inferencia, en UTC (ISO 8601)")
    explicacion: Optional[Explicacion] = None
    aviso: str