import pandas as pd
import numpy as np
from PIL import Image
import torch

from src.dataset import LesionDataset
from src.experimento_class_weight import (
    _guardar_checkpoint,
    metricas,
    preparar_particiones,
    subconjunto_balanceado,
)


def test_particiones_excluyen_test_y_evitan_duplicar_lesiones(tmp_path):
    filas = []
    for clase, prefijo in [("mel", "m"), ("nv", "n")]:
        for i in range(20):
            lesion = f"{prefijo}{i}"
            filas.append(
                {
                    "image_id": f"{lesion}_a",
                    "lesion_id": lesion,
                    "dx": clase,
                }
            )
            filas.append(
                {
                    "image_id": f"{lesion}_b",
                    "lesion_id": lesion,
                    "dx": clase,
                }
            )
    metadata_path = tmp_path / "metadata.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame(filas).to_csv(metadata_path, index=False)
    pd.DataFrame(
        [{"image_id": "m0_a", "lesion_id": "m0", "dx": "mel"},
         {"image_id": "n0_a", "lesion_id": "n0", "dx": "nv"}]
    ).to_csv(test_path, index=False)

    train, val = preparar_particiones(metadata_path, test_path, validation_size=0.25)

    assert set(train.lesion_id).isdisjoint(val.lesion_id)
    assert not {"m0", "n0"} & set(train.lesion_id)
    assert not {"m0", "n0"} & set(val.lesion_id)
    assert not train.lesion_id.duplicated().any()
    assert not val.lesion_id.duplicated().any()
    assert train.etiqueta.value_counts().to_dict() == {0: 14, 1: 14}
    assert val.etiqueta.value_counts().to_dict() == {0: 5, 1: 5}


def test_subconjunto_balanceado_reduce_solo_clase_mayoritaria():
    train = pd.DataFrame(
        {
            "lesion_id": [f"m{i}" for i in range(3)] + [f"n{i}" for i in range(9)],
            "etiqueta": [1] * 3 + [0] * 9,
        }
    )

    resultado = subconjunto_balanceado(train)

    assert resultado.etiqueta.value_counts().to_dict() == {0: 3, 1: 3}
    assert set(resultado.lesion_id) <= set(train.lesion_id)


def test_dataset_carga_desde_directorio_alternativo(tmp_path):
    Image.new("RGB", (8, 8), color="white").save(tmp_path / "imagen.jpg")
    df = pd.DataFrame({"image_id": ["imagen"], "etiqueta": [0]})

    imagen, etiqueta = LesionDataset(df, image_dir=tmp_path)[0]

    assert imagen.size == (8, 8)
    assert etiqueta.item() == 0


def test_metricas_estima_ppv_a_prevalencia_del_diez_por_ciento():
    resultado = metricas(
        np.array([1] * 10 + [0] * 10),
        np.array([0.9] * 8 + [0.1] * 2 + [0.1] * 9 + [0.9]),
        umbral=0.5,
    )

    assert resultado["recall_melanoma"] == 0.8
    assert resultado["especificidad"] == 0.9
    assert abs(resultado["ppv_estimado_prevalencia_10pct"] - 0.470588) < 1e-5


def test_guardar_checkpoint_escribe_estado_en_ruta_separada(tmp_path):
    ruta = tmp_path / "modelos" / "class_weight.pt"
    estado = {"peso": torch.tensor([1.0, 2.0])}

    _guardar_checkpoint(estado, ruta)

    guardado = torch.load(ruta, weights_only=True)
    assert torch.equal(guardado["peso"], estado["peso"])
    assert not ruta.with_suffix(".pt.tmp").exists()
