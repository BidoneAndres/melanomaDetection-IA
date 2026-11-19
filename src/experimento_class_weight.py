import copy
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.dataset import LesionDataset, get_transforms
from src.entrenamiento import entrenar_epoca, evaluar, probabilidades
from src.modelo import crear_modelo, congelar_base, descongelar_ultimo_bloque
from src.utils import SEED, fijar_semillas

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
IMAGES = RAW / "images"
UMBRAL_RECALL = 0.85
UMBRALES = (0.4, 0.5, 0.6, 0.7, 0.8)


def preparar_particiones(metadata_path, test_path, validation_size=0.2, seed=SEED):
    metadata = pd.read_csv(metadata_path)
    test_lesions = set(pd.read_csv(test_path)["lesion_id"])
    datos = metadata[metadata["dx"].isin(["nv", "mel"])].copy()
    datos = datos[~datos["lesion_id"].isin(test_lesions)]
    datos = (
        datos.groupby("lesion_id", group_keys=False)
        .sample(n=1, random_state=seed)
        .reset_index(drop=True)
    )
    datos["etiqueta"] = (datos["dx"] == "mel").astype(int)
    datos["label"] = datos["dx"].map({"nv": "lunar_benigno", "mel": "melanoma"})

    train_df, val_df = train_test_split(
        datos,
        test_size=validation_size,
        stratify=datos["etiqueta"],
        random_state=seed,
    )
    if set(train_df["lesion_id"]) & test_lesions:
        raise ValueError("El entrenamiento contiene lesiones del test anterior.")
    if set(val_df["lesion_id"]) & test_lesions:
        raise ValueError("La validación contiene lesiones del test anterior.")
    if set(train_df["lesion_id"]) & set(val_df["lesion_id"]):
        raise ValueError("Hay lesiones compartidas entre train y validación.")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def subconjunto_balanceado(train_df, seed=SEED):
    cantidad = int(train_df["etiqueta"].value_counts().min())
    return (
        train_df.groupby("etiqueta", group_keys=False)
        .sample(n=cantidad, random_state=seed)
        .reset_index(drop=True)
    )


def seleccionar_umbral(y_true, probabilidades_modelo):
    candidatos = []
    for umbral in UMBRALES:
        predicciones = (probabilidades_modelo >= umbral).astype(int)
        recall = recall_score(y_true, predicciones, zero_division=0)
        if recall >= UMBRAL_RECALL:
            candidatos.append(
                (
                    f1_score(y_true, predicciones, average="macro"),
                    umbral,
                )
            )
    if not candidatos:
        return None
    return max(candidatos, key=lambda candidato: candidato[0])[1]


def metricas(y_true, probabilidades_modelo, umbral):
    predicciones = (probabilidades_modelo >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicciones, labels=[0, 1]).ravel()
    recall = float(recall_score(y_true, predicciones, zero_division=0))
    especificidad = float(tn / (tn + fp)) if tn + fp else 0.0
    prevalencia_referencia = 0.10
    denominador_ppv = (
        recall * prevalencia_referencia
        + (1 - especificidad) * (1 - prevalencia_referencia)
    )
    return {
        "umbral": float(umbral),
        "accuracy": float(accuracy_score(y_true, predicciones)),
        "precision_melanoma": float(
            precision_score(y_true, predicciones, zero_division=0)
        ),
        "recall_melanoma": recall,
        "especificidad": especificidad,
        "ppv_estimado_prevalencia_10pct": (
            float(recall * prevalencia_referencia / denominador_ppv)
            if denominador_ppv
            else 0.0
        ),
        "f1_macro": float(f1_score(y_true, predicciones, average="macro")),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "matriz_confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def _crear_loaders(train_df, val_df, batch_size, seed):
    generador = torch.Generator().manual_seed(seed)
    loader_train = DataLoader(
        LesionDataset(train_df, get_transforms(entrenamiento=True), image_dir=IMAGES),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generador,
    )
    loader_val = DataLoader(
        LesionDataset(val_df, get_transforms(entrenamiento=False), image_dir=IMAGES),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return loader_train, loader_val


def entrenar_modelo(
    train_df,
    val_df,
    ponderar_clases,
    device,
    checkpoint_path=None,
    batch_size=32,
    epocas_fase_1=8,
    epocas_fase_2=12,
    paciencia=4,
    seed=SEED,
):
    fijar_semillas(seed)
    loader_train, loader_val = _crear_loaders(train_df, val_df, batch_size, seed)
    modelo = crear_modelo().to(device)
    congelar_base(modelo)

    if ponderar_clases:
        cantidades = train_df["etiqueta"].value_counts()
        pos_weight = torch.tensor(
            [cantidades[0] / cantidades[1]], dtype=torch.float32, device=device
        )
    else:
        pos_weight = None
    criterio = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    mejor_loss = float("inf")
    mejor_estado = None
    historial = []

    optimizador = torch.optim.Adam(
        [p for p in modelo.parameters() if p.requires_grad], lr=1e-3
    )
    for epoca in range(1, epocas_fase_1 + 1):
        loss_train = entrenar_epoca(
            modelo, loader_train, criterio, optimizador, device
        )
        val = evaluar(modelo, loader_val, criterio, device)
        historial.append(
            {"fase": 1, "epoca": epoca, "loss_train": loss_train, **val}
        )
        print(
            f"[{'ponderado' if ponderar_clases else 'balanceado'} F1] "
            f"época {epoca}/{epocas_fase_1}: train={loss_train:.4f}, "
            f"val={val['loss']:.4f}"
        )
        if val["loss"] < mejor_loss:
            mejor_loss = val["loss"]
            mejor_estado = copy.deepcopy(modelo.state_dict())
            if checkpoint_path is not None:
                _guardar_checkpoint(mejor_estado, checkpoint_path)

    modelo.load_state_dict(mejor_estado)
    descongelar_ultimo_bloque(modelo)
    optimizador = torch.optim.Adam(
        [p for p in modelo.parameters() if p.requires_grad], lr=1e-4
    )
    sin_mejora = 0
    for epoca in range(1, epocas_fase_2 + 1):
        loss_train = entrenar_epoca(
            modelo, loader_train, criterio, optimizador, device
        )
        val = evaluar(modelo, loader_val, criterio, device)
        historial.append(
            {"fase": 2, "epoca": epoca, "loss_train": loss_train, **val}
        )
        print(
            f"[{'ponderado' if ponderar_clases else 'balanceado'} F2] "
            f"época {epoca}/{epocas_fase_2}: train={loss_train:.4f}, "
            f"val={val['loss']:.4f}"
        )
        if val["loss"] < mejor_loss:
            mejor_loss = val["loss"]
            mejor_estado = copy.deepcopy(modelo.state_dict())
            if checkpoint_path is not None:
                _guardar_checkpoint(mejor_estado, checkpoint_path)
            sin_mejora = 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                print("Parada temprana.")
                break

    modelo.load_state_dict(mejor_estado)
    probabilidades_val, y_true = probabilidades(modelo, loader_val, device)
    return {
        "modelo": modelo,
        "y_true": y_true,
        "probabilidades": probabilidades_val,
        "historial": historial,
        "pos_weight": float(pos_weight.item()) if pos_weight is not None else None,
    }


def _guardar_checkpoint(estado, ruta):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    torch.save(estado, temporal)
    temporal.replace(ruta)


def ejecutar_experimento():
    train_df, val_df = preparar_particiones(
        RAW / "HAM10000_metadata.csv", PROCESSED / "test.csv"
    )
    train_balanceado = subconjunto_balanceado(train_df)
    prevalencia_validacion = float(val_df["etiqueta"].mean())
    print(
        f"Train sin test previo: {len(train_df)} imágenes "
        f"({int(train_df.etiqueta.sum())} melanomas, "
        f"{int((train_df.etiqueta == 0).sum())} nevos)"
    )
    print(
        f"Validación: {len(val_df)} imágenes, "
        f"prevalencia de melanoma {prevalencia_validacion:.1%}"
    )
    print(f"Control submuestreado: {len(train_balanceado)} imágenes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo:", device)
    resultados = {}
    for nombre, datos_train, ponderar in [
        ("class_weight", train_df, True),
        ("submuestreo", train_balanceado, False),
    ]:
        salida = entrenar_modelo(
            datos_train,
            val_df,
            ponderar_clases=ponderar,
            device=device,
        )
        resultados[nombre] = {
            "pos_weight": salida["pos_weight"],
            "umbral_0_5": metricas(
                salida["y_true"], salida["probabilidades"], 0.5
            ),
            "umbral_seleccionado": None,
            "historial": salida["historial"],
        }
        umbral = seleccionar_umbral(salida["y_true"], salida["probabilidades"])
        if umbral is not None:
            resultados[nombre]["umbral_seleccionado"] = metricas(
                salida["y_true"], salida["probabilidades"], umbral
            )
        else:
            print(
                f"{nombre}: ningún umbral de {UMBRALES} alcanza "
                f"recall >= {UMBRAL_RECALL:.2f} en validación."
            )
        del salida
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    reporte = {
        "experimento": "class_weight vs. submuestreo",
        "evaluacion": "validacion exploratoria; no se uso el test anterior",
        "semilla": SEED,
        "lesiones_test_anterior_excluidas": int(
            pd.read_csv(PROCESSED / "test.csv")["lesion_id"].nunique()
        ),
        "entrenamiento_class_weight": {
            "imagenes": len(train_df),
            "melanomas": int(train_df["etiqueta"].sum()),
            "nevos": int((train_df["etiqueta"] == 0).sum()),
        },
        "entrenamiento_submuestreado": {
            "imagenes": len(train_balanceado),
            "melanomas": int(train_balanceado["etiqueta"].sum()),
            "nevos": int((train_balanceado["etiqueta"] == 0).sum()),
        },
        "tamano_validacion": len(val_df),
        "prevalencia_melanoma_validacion": prevalencia_validacion,
        "prevalencia_referencia_ppv": 0.10,
        "entrenamiento": {
            "modelo": "ResNet18 preentrenada en ImageNet",
            "fase_1_epocas_maximas": 8,
            "fase_2_epocas_maximas": 12,
            "paciencia": 4,
            "batch_size": 32,
        },
        "regla_umbral": (
            f"grilla {UMBRALES}; elegir mayor F1 macro entre "
            f"umbrales con recall >= {UMBRAL_RECALL:.2f}"
        ),
        "resultados": resultados,
    }
    report_path = ROOT / "reports" / "class_weight_experiment.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(reporte, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(reporte, indent=2, ensure_ascii=False))
    return reporte


if __name__ == "__main__":
    ejecutar_experimento()
