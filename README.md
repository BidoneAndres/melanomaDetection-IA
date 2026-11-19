<div align="center">

# clasificacion-melanoma

### Sistema de apoyo a la decisión que estima la probabilidad de que una lesión cutánea sea un melanoma o un nevo benigno

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-en%20desarrollo-orange.svg)]()

[Reportar un bug](../../issues) · [Pedir una funcionalidad](../../issues)

</div>

---

## 📋 Tabla de contenidos

- [Demo](#-demo)
- [Sobre el proyecto](#-sobre-el-proyecto)
  - [Contexto académico](#contexto-académico)
  - [Construido con](#construido-con)
- [El dataset](#-el-dataset)
- [Metodología](#-metodología)
- [Resultados](#-resultados)
- [Limitaciones y alcance](#-limitaciones-y-alcance)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Uso](#-uso)
- [Roadmap](#-roadmap)
- [Autores](#-autores)
- [Licencia](#-licencia)
- [Agradecimientos y referencias](#-agradecimientos-y-referencias)

---

## 🎬 Demo

La demo interactiva publicada en GitHub Pages está en desarrollo (ver [Roadmap](#-roadmap)). En esta iteración (MVP) el sistema se usa por consola: recibe una imagen dermatoscópica y devuelve un JSON con la probabilidad de melanoma.

```json
{
  "archivo": "imagen.jpg",
  "probabilidad_melanoma": 0.1234,
  "etiqueta": "nevo",
  "umbral": 0.7
}
```

**🔗 Demo en vivo:** pendiente de publicación.

---

## 📖 Sobre el proyecto

El diagnóstico dermatológico inicial presenta variabilidad entre profesionales, y una lesión maligna puede pasar desapercibida. El melanoma es un tipo de cáncer de piel que puede aparecer como un lunar nuevo o como un lunar existente que cambia, y su detección temprana es determinante.

Este proyecto desarrolla un clasificador binario (**melanoma** vs. **nevo**) para imágenes dermatoscópicas. Dada una imagen, el sistema devuelve la probabilidad de melanoma y la etiqueta resultante, como herramienta de apoyo a la decisión. 

Se comparan dos enfoques: un baseline tradicional (Random Forest sobre características geométricas y de color extraídas con OpenCV) y una red convolucional (ResNet18 con transfer learning). Ambos superan las metas fijadas (recall de melanoma y F1 macro mayores al 80 %). La comparación pareada actual es exploratoria porque el conjunto de prueba ya se había consultado en una evaluación anterior.

### Contexto académico

- **Asignatura:** Inteligencia Artificial
- **Institución:** Universidad de la Defensa Nacional, Centro Regional Universitario Córdoba (IUA), Facultad de Ingeniería, Ingeniería en Informática
- **Cohorte / año:** 2026
- **Docente:** Ing. Hernando Alexis González

### Construido con

- [Python 3.11](https://www.python.org/)
- [PyTorch](https://pytorch.org/) / [torchvision](https://pytorch.org/vision/) — red convolucional y transfer learning
- [OpenCV](https://opencv.org/) — preprocesamiento, segmentación y extracción de características
- [scikit-learn](https://scikit-learn.org/) — baseline y métricas
- [pandas](https://pandas.pydata.org/) / [NumPy](https://numpy.org/) / [matplotlib](https://matplotlib.org/)
- [Docker](https://www.docker.com/) — entorno reproducible

---

## 📊 El dataset

| Campo | Detalle |
|---|---|
| Fuente | [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) (Tschandl et al., 2018) |
| Registros | 10015 imágenes dermatoscópicas de 7 tipos de lesión; se usan solo melanoma (`mel`) y nevo (`nv`) |
| Tamaño de imagen | 600 x 450 píxeles |
| Variables | `image_id`, `lesion_id`, `dx` (diagnóstico), edad, sexo, localización |
| Variable objetivo | `dx`, convertida a `etiqueta` (1 = melanoma, 0 = nevo) |
| Origen | Dos centros (Viena y Australia); predominan pieles claras |
| Licencia del dataset | Consultar la página de origen antes de redistribuir |

**Decisiones sobre los datos:**

- Se conservaron solo las lesiones con **una única imagen** (`lesion_id` único), para evitar que fotos de la misma lesión caigan en train y test (fuga de datos).
- En el experimento original se balanceó por submuestreo **después** de filtrar. Esto descartó cerca del 80 % de los nevos disponibles y dejó una prevalencia artificial de melanoma del 50 %; por eso, las métricas de precisión del test balanceado no representan directamente una población real.
- La partición train/test se hizo antes de cualquier preprocesamiento, de forma estratificada.

**Volumen final:** 1228 imágenes (614 por clase).

| Conjunto | Imágenes | Uso |
|---|---|---|
| Train | 982 | Validación cruzada del baseline en el notebook 06 |
| Train final | 785 | Entrenamiento de la CNN y del baseline comparativo |
| Validación | 197 | Parada temprana de la CNN y selección de umbral para ambos modelos |
| Test | 246 (123 por clase) | Evaluación final |

Las imágenes originales de HAM10000 y las características extraídas no se versionan. El repositorio sí incluye algunos CSV ligeros de metadatos y particiones.

---

## 🔬 Metodología

### Análisis exploratorio

Todas las imágenes miden 600 x 450. La exploración mostró tres problemas que afectan a la segmentación: pelos que cruzan la lesión, esquinas negras del dermatoscopio y viñeteado (bordes más oscuros). En el baseline, la circularidad, la solidez y la asimetría separan parcialmente las clases: los melanomas tienen bordes más irregulares y asimétricos.

### Preprocesamiento

Para el baseline, cada imagen pasa por este pipeline (`src/preprocesamiento.py`):

1. **Eliminación de pelos:** filtro morfológico black-hat más inpainting.
2. **Máscara de esquinas negras:** zonas oscuras conectadas a las esquinas de la imagen.
3. **Segmentación de la lesión con Otsu**, calculando el umbral solo con los píxeles válidos, con corrección de iluminación, limpieza morfológica y conservando la componente conexa más grande.

La CNN usa las imágenes originales, redimensionadas a 224 x 224 y normalizadas con las estadísticas de ImageNet. En entrenamiento se aplica aumentación (giros, rotaciones y cambios leves de brillo y contraste).

### Modelo(s)

**Baseline:** Random Forest (300 árboles) sobre características geométricas (área, área relativa, perímetro, circularidad, solidez, asimetría) y de color (media y desvío de cada canal en HSV y LAB). Se comparó con regresión logística y con un clasificador "dummy" mediante validación cruzada de 5 particiones en train.

**CNN:** ResNet18 preentrenada en ImageNet, con una salida (logit de melanoma). Entrenamiento en dos fases:

1. Se entrena solo la capa final (tasa de aprendizaje 1e-3, hasta 8 épocas).
2. Ajuste fino del último bloque convolucional (tasa 1e-4, hasta 12 épocas, con parada temprana).

Se guarda el checkpoint con menor pérdida de validación. Se eligió ResNet18 por su tamaño reducido y su ejecución en CPU dado el volumen de datos; **no se compararon otras arquitecturas**.

### Validación

- Split train/test 80/20 estratificado; el test no se usa para seleccionar modelos ni umbrales.
- Baseline: validación cruzada de 5 particiones en las 982 imágenes de train (notebook 06). Para la comparación final, se entrenó con las mismas 785 imágenes que la CNN.
- CNN: 20 % de train (197 imágenes) como validación para parada temprana y selección de umbral. Esa misma validación se usó para el baseline comparativo.
- **Umbral de decisión de ambos modelos:** se usó la misma grilla (0,4 a 0,8, en pasos de 0,1) y el mismo criterio: entre los umbrales con recall de melanoma ≥ 0,85, elegir el de mayor F1 macro. Resultado: 0,7 para la CNN y 0,5 para el baseline.
- Intervalos de confianza al 95 % por bootstrap (2000 remuestreos).
- **Semilla global 42** en todo proceso aleatorio (balanceo, splits, validación cruzada, Random Forest, inicialización, aumentación y bootstrap).

---

## 📈 Resultados

Umbral de la CNN según la validación (197 imágenes):

| Umbral | Recall | F1 macro | Falsos positivos | Falsos negativos |
|---|---|---|---|---|
| 0,4 | 0,959 | 0,814 | 32 | 4 |
| 0,5 | 0,908 | 0,811 | 28 | 9 |
| 0,6 | 0,888 | 0,827 | 23 | 11 |
| **0,7** | **0,857** | **0,848** | **16** | **14** |
| 0,8 | 0,776 | 0,832 | 11 | 22 |

El umbral del baseline seleccionado con la misma regla fue 0,5 (recall 0,898 y F1 macro 0,795 en validación).

Comparación exploratoria en test (246 imágenes, 123 melanomas y 123 nevos; ambos modelos sobre las mismas imágenes):

| Modelo | Recall melanoma (IC 95 %) | F1 macro (IC 95 %) | Falsos negativos | Falsos positivos |
|---|---|---|---|---|
| Baseline (RF + OpenCV) | 0,878 [0,82, 0,94] | 0,833 [0,78, 0,88] | 15 | 26 |
| CNN (ResNet18) | 0,902 [0,85, 0,95] | 0,886 [0,84, 0,93] | 12 | 16 |

Bootstrap pareado de la diferencia (CNN menos baseline):

| Métrica | Diferencia | IC 95 % pareado |
|---|---:|---:|
| Recall melanoma | 0,024 | [-0,050, 0,098] |
| F1 macro | 0,053 | [0,008, 0,098] |

El intervalo pareado del recall incluye cero; el de F1 macro no. Sin embargo, **el test ya se había consultado en una evaluación anterior**, por lo que esta comparación es exploratoria y no constituye una confirmación independiente de superioridad. Ambos modelos superan el 80 % en recall y F1 macro en esta muestra.

El test tiene prevalencia de melanoma del 50 %, mientras que una prevalencia de referencia más realista ronda el 10 %. Para ilustrar el efecto, se estimó el valor predictivo positivo (VPP) con la sensibilidad y especificidad observadas en esta comparación, suponiendo que ambas se mantienen con esa prevalencia:

| Modelo | VPP estimado con prevalencia de melanoma del 10 % |
|---|---:|
| Baseline (RF + OpenCV) | 31,6 % |
| CNN (ResNet18) | 43,5 % |

Estos VPP son **estimaciones ajustadas**, no resultados medidos en una muestra con prevalencia real. Los falsos positivos reportados arriba son conteos sobre los 123 nevos del test balanceado, no una estimación directa de la cantidad esperable en una población. El cálculo no reemplaza una evaluación externa con prevalencia representativa.

### Experimento exploratorio con `class_weight`

Se compararon dos ResNet18, una con `class_weight` (`pos_weight = 10,746`) y otra entrenada con submuestreo. Ambas usaron la misma validación de 1155 imágenes (8,5 % de melanomas). Se excluyeron por `lesion_id` las 246 lesiones del test consultado anteriormente; ese test no se evaluó. El entrenamiento ponderado usó 4616 imágenes (393 melanomas y 4223 nevos), mientras que el control submuestreado usó 786 (393 de cada clase).

Para cada modelo se seleccionó el umbral con la misma regla: en la grilla [0,4; 0,5; 0,6; 0,7; 0,8], elegir el de mayor F1 macro entre los que alcanzan recall de melanoma ≥ 0,85. Los umbrales elegidos fueron 0,5 para `class_weight` y 0,6 para el submuestreo:

| Modelo | Umbral | Recall melanoma | F1 macro | Especificidad | VPP estimado al 10 % | Falsos negativos | Falsos positivos |
|---|---:|---:|---:|---:|---:|---:|---:|
| `class_weight` | 0,5 | 0,878 | 0,720 | 0,861 | 41,2 % | 12 | 147 |
| Submuestreo | 0,6 | 0,857 | 0,696 | 0,843 | 37,8 % | 14 | 166 |

Bootstrap pareado de la diferencia (`class_weight` menos submuestreo):

| Métrica | Diferencia observada | IC 95 % pareado |
|---|---:|---:|
| Recall de melanoma | 0,020 | [-0,043; 0,083] |
| F1 macro | 0,024 | [0,0003; 0,048] |

El IC del recall incluye cero; el de F1 macro apenas queda por encima. La validación se usó para parada temprana y selección de umbral, por lo que estos resultados son exploratorios y no confirman superioridad. Tampoco se comparan directamente con las métricas de la CNN original, que se obtuvieron en otra partición.

### Subgrupos y ablación

Se evaluaron los dos modelos experimentales en subgrupos de edad usando los umbrales globales seleccionados. Se excluyeron las 9 lesiones sin edad. Los tamaños pequeños, especialmente en melanomas menores de 50 años, limitan la interpretación:

| Modelo | Edad | Melanomas/nevos | Recall melanoma | Especificidad nevo | F1 macro |
|---|---|---:|---:|---:|---:|
| `class_weight` | <50 | 25/554 | 0,800 | 0,888 | 0,655 |
| `class_weight` | ≥50 | 72/495 | 0,903 | 0,832 | 0,746 |
| Submuestreo | <50 | 25/554 | 0,680 | 0,856 | 0,597 |
| Submuestreo | ≥50 | 72/495 | 0,917 | 0,830 | 0,748 |

En esta validación el recall fue menor para menores de 50 años en ambos modelos; este resultado es descriptivo y no demuestra que la edad cause la diferencia.

La fuente del diagnóstico quedó confundida con la clase: **los 98 melanomas son histopatológicos**, mientras que seguimiento y consenso solo contienen nevos. Por eso, recall y F1 no se pueden estimar para esos dos últimos grupos; allí se informa únicamente la especificidad de nevos:

| Modelo | Fuente | Melanomas/nevos | Recall melanoma | Especificidad nevo | F1 macro |
|---|---|---:|---:|---:|---:|
| `class_weight` | Histopatología | 98/260 | 0,878 | 0,550 | 0,630 |
| `class_weight` | Seguimiento | 0/726 | — | 0,989 | — |
| `class_weight` | Consenso | 0/71 | — | 0,690 | — |
| Submuestreo | Histopatología | 98/260 | 0,857 | 0,477 | 0,576 |
| Submuestreo | Seguimiento | 0/726 | — | 0,996 | — |
| Submuestreo | Consenso | 0/71 | — | 0,620 | — |

Esta composición impide descartar que la fuente del diagnóstico esté asociada a la clase en estos datos.

Para la ablación del baseline se entrenaron Random Forest con características de forma, de color y ambas, usando 785 imágenes de train y 197 de validación. El test no se usó. Se aplicó la misma regla de selección de umbral; para forma no hubo umbral que alcanzara recall ≥ 0,85, así que se muestra su resultado con umbral 0,5:

| Características | Umbral | Recall melanoma | F1 macro | Especificidad | Falsos negativos | Falsos positivos |
|---|---:|---:|---:|---:|---:|---:|
| Solo forma (6) | 0,5 (referencia) | 0,561 | 0,632 | 0,707 | 43 | 29 |
| Solo color (12) | 0,6 | 0,857 | 0,786 | 0,717 | 14 | 28 |
| Forma y color (18) | 0,5 | 0,898 | 0,795 | 0,697 | 10 | 30 |

En esta validación las características de color rindieron mejor que las de forma por sí solas. Combinar forma y color elevó el recall, pero no la especificidad. Son resultados exploratorios de una única partición y no prueban causalidad ni generalización.

Las características del baseline se regeneraron con el pipeline final antes de esta comparación. Debido a la evaluación previa del test, los nuevos resultados se reportan como exploratorios.

**Análisis crítico:** el baseline depende mucho del color (la característica más importante del Random Forest es `lab2_media`), lo que podría reflejar una pista de la cámara o la iluminación y no de la lesión; la ablación es consistente con esa posibilidad, pero no la confirma. La CNN muestra un comportamiento más equilibrado, aunque no se verificó en qué se apoya.

---

## ⚠️ Limitaciones y alcance

**Limitaciones:**

- Test de 246 imágenes (123 melanomas): los resultados tienen un margen de error de varios puntos.
- El submuestreo produjo un test balanceado que no refleja la prevalencia poblacional; el VPP ajustado al 10 % es una estimación y depende de que sensibilidad y especificidad se mantengan.
- La comparación entre `class_weight` y submuestreo comparte validación, pero esta se usó para parada temprana y selección de umbral; los resultados son exploratorios, no una evaluación independiente.
- En HAM10000, la fuente histopatológica aparece en todos los melanomas de la validación; no hay melanomas de seguimiento o consenso para una comparación equivalente por fuente.
- Los análisis por edad y la ablación usan validaciones pequeñas y no sustituyen una evaluación independiente.
- No se realizó una evaluación con imágenes de una fuente externa; queda fuera del alcance acordado para esta etapa.
- El filtro por `lesion_id` único elimina las lesiones con varias fotos, que suelen ser las más dudosas, por lo que probablemente simplifica el problema.
- Posible dependencia del color en el baseline (ver análisis crítico).
- El modelo solo es válido para imágenes dermatoscópicas similares a las de HAM10000; no generaliza a fotos comunes.
- Para esta comparación ambos modelos se entrenaron con 785 imágenes; la validación cruzada descriptiva del notebook 06 usa las 982 imágenes de train.
- La segmentación se evaluó visualmente, sin máscaras de referencia.
- Los resultados pueden variar levemente entre sistemas por diferencias numéricas de las librerías, aun con semillas fijas.
- La probabilidad es una salida del modelo y **no es un diagnóstico médico**.

**Fuera de alcance de esta iteración (MVP):**

- Clasificación de las otras 5 lesiones de HAM10000.
- Imágenes no dermatoscópicas.
- Autenticación de usuarios, base de datos de historiales e interfaz de usuario.
- Validación clínica o uso como dispositivo médico.

---

## 📂 Estructura del proyecto

```text
.
├── README.md
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt          # inferencia (versiones fijas)
├── imagenesProbar/           # ejemplos de melanoma y nevo
├── src/
│   ├── preprocesamiento.py   # pelos, esquinas y segmentación
│   ├── dataset.py            # Dataset y aumentación (PyTorch)
│   ├── modelo.py             # ResNet18 y congelado de capas
│   ├── entrenamiento.py      # bucles de entrenamiento y evaluación
│   ├── inferencia.py         # predicción sobre una imagen
│   ├── predecir.py           # script de consola
│   ├── division.py           # partición estratificada train/test
│   └── utils.py              # semillas
├── notebooks/                # exploración y experimentos (01 a 12)
├── tests/                    # pruebas del split y la inferencia
├── data/                     # imágenes y características generadas, excluidas de Git
│   ├── raw/                  # HAM10000_metadata.csv versionado
│   └── processed/            # metadatos y particiones CSV versionados
└── models/
    └── cnn_resnet18.pt       # checkpoint incluido en el repositorio (~44 MB)
```

---

## ⚙️ Instalación

### Prerrequisitos

- [Docker](https://docs.docker.com/get-docker/) con Docker Compose
- Git
- El checkpoint `models/cnn_resnet18.pt` está incluido en el repositorio para permitir la inferencia sin reentrenar. El notebook 08 permite volver a generarlo.
- El notebook 10 entrena una variante experimental con `class_weight` y guarda un checkpoint separado (`models/cnn_resnet18_class_weight.pt`); no reemplaza el modelo usado por inferencia.
- Para reproducir el flujo completo, descargar y extraer HAM10000 desde [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?resource=download). Dejar `HAM10000_metadata.csv` en `data/raw/` y copiar las imágenes de `HAM10000_images_part_1/` y `HAM10000_images_part_2/` a una única carpeta `data/raw/images/`, conservando los nombres originales.

### Pasos

Crear la carpeta con las imágenes después de extraer los dos archivos ZIP. En PowerShell:

```powershell
New-Item -ItemType Directory -Force data/raw/images
Get-ChildItem -Path data/raw/HAM10000_images_part_1,data/raw/HAM10000_images_part_2 -Filter *.jpg |
    Copy-Item -Destination data/raw/images
```

En Linux o macOS:

```bash
mkdir -p data/raw/images
cp data/raw/HAM10000_images_part_1/*.jpg data/raw/HAM10000_images_part_2/*.jpg data/raw/images/
```

La construcción de la imagen Docker no necesita que HAM10000 esté descargado; los datos se montan desde el equipo anfitrión al ejecutar los notebooks.

```bash
git clone https://github.com/BidoneAndres/melanomaDetection-IA
cd melanomaDetection-IA
docker compose build
```

La primera construcción tarda por la descarga de las dependencias; las siguientes reutilizan la caché de Docker.

### Instalar dependencias y ejecutar pruebas localmente

Con [uv](https://docs.astral.sh/uv/), instalar el entorno del proyecto y las dependencias de desarrollo:

```bash
uv sync --extra dev
uv run pytest
```

También se pueden instalar las dependencias de ejecución directamente desde `requirements.txt`:

```bash
uv venv
uv pip install --python .venv -r requirements.txt
```


## ▶️ Uso

### Clasificar una imagen

La carpeta `imagenesProbar/` incluye ejemplos de ambas clases (`melanoma1.jpg` a `melanoma4.jpg` y `nevo1.jpg`). También se pueden colocar otras imágenes dermatoscópicas allí:

```bash
docker compose run --build --rm app python -m src.predecir /app/imagenesProbar/imagen.jpg
```

El programa imprime un JSON con el nombre del archivo, la probabilidad estimada de melanoma, la etiqueta asignada y el umbral utilizado. Para guardar una copia etiquetada:

```bash
docker compose run --build --rm app python -m src.predecir /app/imagenesProbar/imagen.jpg --guardar /app/imagenesProbar/prediccion.jpg
```

La inferencia usa `models/cnn_resnet18.pt` y el umbral configurado en `src/inferencia.py`. 
Si el archivo no existe o no es una imagen válida, el programa devuelve un JSON con el error y termina con código distinto de cero.

### Reproducir el flujo de trabajo

Construir la imagen e iniciar Jupyter en segundo plano:

```bash
docker compose up --build -d
docker compose logs app
```

Abrir http://localhost:8888 e ingresar el token que aparece en los logs. Ejecutar los notebooks en este orden:

1. `01-FiltradoYBalanceoDeDatos.ipynb`: filtra HAM10000, conserva nevo y melanoma, y prepara los datos balanceados en `data/processed/`.
2. `02-SplitEnTrainYTest.ipynb`: crea las particiones de entrenamiento y prueba.
3. `03-ExploracionDeImagenes.ipynb` y `04-Preprocesamiento.ipynb`: exploran las imágenes y muestran el preprocesamiento.
4. `05-EstraccionDeCaracteristicas.ipynb`: genera las características para el baseline.
5. `06-BaselineYEvaluacion.ipynb`: entrena y evalúa los modelos baseline.
6. `07-DataSetRed.ipynb`: prepara las particiones de entrenamiento y validación de la CNN.
7. `08-EntrenamientoCNN.ipynb`: entrena la CNN y guarda `models/cnn_resnet18.pt`.
8. `09-EvaluacionYComparacion.ipynb`: elige el umbral con validación y compara los modelos en test.
9. `10-Mejoras.ipynb` (experimento opcional): entrena la variante con `class_weight` y guarda un checkpoint separado. Usa una partición distinta, excluye el test anterior y no reemplaza el modelo de inferencia.
10. `11-ComparacionClassWeight.ipynb`: compara `class_weight` con el control submuestreado en la misma validación; reutiliza el checkpoint del notebook 10 y guarda el control y el reporte de la comparación.
11. `12-SubgruposYAblacion.ipynb`: analiza los dos modelos por edad y fuente del diagnóstico, y compara el baseline con características de forma, color y ambas; no usa el test.

Ejecutar de nuevo los notebooks de preparación regenera los archivos de `data/processed/`, incluida la partición de prueba. Para detener el contenedor:

```bash
docker compose down
```

---

## 🗺️ Roadmap

- [ ] API REST con FastAPI (validación de entrada y errores HTTP claros)
- [ ] Interfaz de usuario y demo publicada en GitHub Pages
- [x] Pruebas automatizadas básicas con `pytest` para el split y la inferencia
- [ ] Logging de la aplicación
- [x] Entrenamiento experimental de CNN con `class_weight`
- [x] Comparación directa entre `class_weight` y submuestreo con las mismas particiones
- [x] Análisis exploratorio por subgrupos de edad y fuente del diagnóstico
- [x] Ablación del baseline (solo forma, solo color y ambas)
- [ ] Comparación con otras arquitecturas

---

## 👥 Autores

- **Bidone Andres** — [GitHub](https://github.com/BidoneAndres)

---

## 📄 Licencia

Este proyecto está bajo la licencia [MIT](LICENSE) — ver el archivo `LICENSE` para más detalle. El dataset HAM10000 tiene sus propias condiciones de uso, indicadas en su página de origen.

---

## 🙏 Agradecimientos y referencias

- Tschandl, P., Rosendahl, C. y Kittler, H. (2018). *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data, 5, 180161.
- He, K., Zhang, X., Ren, S. y Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
- Otsu, N. (1979). *A threshold selection method from gray-level histograms.* IEEE Transactions on Systems, Man, and Cybernetics.
- Docente: Ing. Hernando Alexis González, por los lineamientos del proyecto.