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

**🔗 Demo en vivo:** https://BidoneAndres.github.io/melanomaDetection-IA/ *(pendiente de publicación)*

---

## 📖 Sobre el proyecto

El diagnóstico dermatológico inicial presenta variabilidad entre profesionales, y una lesión maligna puede pasar desapercibida. El melanoma es un tipo de cáncer de piel que puede aparecer como un lunar nuevo o como un lunar existente que cambia, y su detección temprana es determinante.

Este proyecto desarrolla un clasificador binario (**melanoma** vs. **nevo**) para imágenes dermatoscópicas. Dada una imagen, el sistema devuelve la probabilidad de melanoma y la etiqueta resultante, como herramienta de apoyo a la decisión. 

Se comparan dos enfoques: un baseline tradicional (Random Forest sobre características geométricas y de color extraídas con OpenCV) y una red convolucional (ResNet18 con transfer learning). Ambos superan las metas fijadas (recall de melanoma y F1 macro mayores al 80 %), pero con el tamaño del conjunto de prueba **no se puede afirmar que la CNN supere al baseline**; su diferencia más clara está en los falsos positivos.

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
- Se balanceó por submuestreo **después** de filtrar, ya que el original contiene muchos más nevos que melanomas y el modelo tendería a predecir siempre "nevo" (falsos negativos).
- La partición train/test se hizo antes de cualquier preprocesamiento, de forma estratificada.

**Volumen final:** 1228 imágenes (614 por clase).

| Conjunto | Imágenes | Uso |
|---|---|---|
| Train | 982 | Baseline (validación cruzada) |
| Train de la CNN | 785 | Entrenamiento de la red |
| Validación | 197 | Parada temprana y elección del umbral (solo CNN) |
| Test | 246 (123 por clase) | Evaluación final |

Los datos no se versionan (`data/` está excluido del repositorio).

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

- Split train/test 80/20 estratificado; el test se usa solo para la evaluación final.
- Baseline: validación cruzada de 5 particiones, solo en train.
- CNN: 20 % de train como validación, para la parada temprana y para elegir el umbral.
- **Umbral de decisión:** se fijó antes de evaluar en test, con la regla "entre los umbrales con recall de melanoma ≥ 0,85, el de mayor F1 macro" (resultado: 0,7).
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

Evaluación final en test (246 imágenes, 123 melanomas y 123 nevos; ambos modelos sobre las mismas imágenes):

| Modelo | Recall melanoma (IC 95 %) | F1 macro (IC 95 %) | Falsos negativos | Falsos positivos |
|---|---|---|---|---|
| Baseline (RF + OpenCV) | 0,886 [0,83, 0,94] | 0,841 [0,79, 0,89] | 14 | 25 |
| CNN (ResNet18) | 0,902 [0,85, 0,95] | 0,886 [0,84, 0,93] | 12 | 16 |

Ambos modelos superan las metas del 80 %. Los intervalos de confianza se solapan, por lo que **con 123 casos por clase no se puede afirmar que la CNN supere al baseline**. La diferencia más clara está en los falsos positivos (16 contra 25), con un recall equivalente.

La CNN se evaluó en test una única vez. Las características del baseline se regeneraron con el pipeline final antes de su evaluación definitiva.

**Análisis crítico:** el baseline depende mucho del color (la característica más importante del Random Forest es `lab2_media`), lo que podría reflejar una pista de la cámara o la iluminación y no de la lesión. La CNN muestra un comportamiento más equilibrado, pero no se verificó en qué se apoya. Con más tiempo se haría un análisis de ablación (solo forma y solo color), se compararían otras arquitecturas y se probaría con una fuente de imágenes distinta a HAM10000.

---

## ⚠️ Limitaciones y alcance

**Limitaciones:**

- Test de 246 imágenes (123 melanomas): los resultados tienen un margen de error de varios puntos.
- El filtro por `lesion_id` único elimina las lesiones con varias fotos, que suelen ser las más dudosas, por lo que probablemente simplifica el problema.
- Posible dependencia del color en el baseline (ver análisis crítico).
- El modelo solo es válido para imágenes dermatoscópicas similares a las de HAM10000; no generaliza a fotos comunes.
- La CNN se entrenó con 785 imágenes y el baseline con 982, por lo que la comparación no es perfectamente pareja.
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
├── pyprojet.toml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt          # inferencia (versiones fijas)
├── imagenesProbar/           # imágenes para probar la inferencia
├── src/
│   ├── preprocesamiento.py   # pelos, esquinas y segmentación
│   ├── dataset.py            # Dataset y aumentación (PyTorch)
│   ├── modelo.py             # ResNet18 y congelado de capas
│   ├── entrenamiento.py      # bucles de entrenamiento y evaluación
│   ├── inferencia.py         # predicción sobre una imagen
│   ├── predecir.py           # script de consola
│   └── utils.py              # semillas
├── notebooks/                # exploración y experimentos (01 a 09)
├── data/                     # excluido de Git
│   ├── raw/                  #   HAM10000 original
│   └── processed/            #   metadatos, splits y características
└── models/                   # pesos y historial de entrenamiento
```

---

## ⚙️ Instalación

### Prerrequisitos

- [Docker](https://docs.docker.com/get-docker/) con Docker Compose
- Git
- Para ejecutar la inferencia se necesita el modelo `models/cnn_resnet18.pt`, que se genera con el notebook 08 (los pesos no se versionan).
- Para reproducir el flujo completo, HAM10000 con `HAM10000_metadata.csv` en `data/raw/` y las imágenes en `data/raw/images/`. Se descarga de [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?resource=download) y las imágenes hay que unificarlas en una sola carpeta `images`.

### Pasos

```bash
git clone https://github.com/BidoneAndres/melanomaDetection-IA
cd melanomaDetection-IA
docker compose build
```

La primera construcción tarda por la descarga de las dependencias; las siguientes reutilizan la caché de Docker.


## ▶️ Uso

### Clasificar una imagen

Colocar las imágenes dermatoscópicas en la carpeta `imagenesProbar/`:

```bash
docker compose run --build --rm app python -m src.predecir /app/imagenesProbar/imagen.jpg
```

El programa imprime un JSON con el nombre del archivo, la probabilidad estimada de melanoma, la etiqueta asignada y el umbral utilizado. Para guardar una copia etiquetada:

```bash
docker compose run --build --rm app python -m src.predecir /app/imagenesProbar/imagen.jpg --guardar /app/imagenesProbar/prediccion.jpg
```

La inferencia usa `models/cnn_resnet18.pt` y el umbral configurado en `src/inferencia.py`. 

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

Ejecutar de nuevo los notebooks de preparación regenera los archivos de `data/processed/`, incluida la partición de prueba. Para detener el contenedor:

```bash
docker compose down
```

---

## 🗺️ Roadmap

- [ ] API REST con FastAPI (validación de entrada y errores HTTP claros)
- [ ] Interfaz de usuario y demo publicada en GitHub Pages
- [ ] Tests automatizados con `pytest` y logging
- [ ] Análisis de ablación (solo forma y solo color) y comparación con otras arquitecturas
- [ ] Evaluación con imágenes dermatoscópicas de una fuente distinta a HAM10000

---

## 👥 Autores

- **Bidone Andres** — [GitHub](https://github.com/BidoneAndres) · [LinkedIn](https://linkedin.com/in/usuario)

---

## 📄 Licencia

Este proyecto está bajo la licencia [MIT](LICENSE) — ver el archivo `LICENSE` para más detalle. El dataset HAM10000 tiene sus propias condiciones de uso, indicadas en su página de origen.

---

## 🙏 Agradecimientos y referencias

- Tschandl, P., Rosendahl, C. y Kittler, H. (2018). *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data, 5, 180161.
- He, K., Zhang, X., Ren, S. y Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
- Otsu, N. (1979). *A threshold selection method from gray-level histograms.* IEEE Transactions on Systems, Man, and Cybernetics.
- Docente: Ing. Hernando Alexis González, por los lineamientos del proyecto.