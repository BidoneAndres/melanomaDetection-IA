import json
import sys

import pytest
from PIL import Image

from src import inferencia
from src.predecir import main


def test_predecir_archivo_invalido_lanza_error_claro(tmp_path):
    ruta = tmp_path / "archivo.jpg"
    ruta.write_text("esto no es una imagen", encoding="utf-8")

    with pytest.raises(ValueError, match="no es una imagen válida"):
        inferencia.predecir(ruta)


def test_predecir_archivo_valido_devuelve_etiqueta(tmp_path, monkeypatch):
    ruta = tmp_path / "nevo.jpg"
    Image.new("RGB", (8, 8), color="white").save(ruta)
    monkeypatch.setattr(inferencia, "predecir_imagen", lambda imagen, umbral: (0.2, "nevo"))

    resultado = inferencia.predecir(ruta)

    assert resultado["archivo"] == "nevo.jpg"
    assert resultado["probabilidad_melanoma"] == 0.2
    assert resultado["etiqueta"] == "nevo"


def test_cli_informa_archivo_invalido_sin_traceback(tmp_path, monkeypatch, capsys):
    ruta = tmp_path / "archivo.jpg"
    ruta.write_text("esto no es una imagen", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["predecir", str(ruta)])

    with pytest.raises(SystemExit) as error:
        main()

    salida = json.loads(capsys.readouterr().out)
    assert error.value.code == 1
    assert "no es una imagen válida" in salida["error"]
