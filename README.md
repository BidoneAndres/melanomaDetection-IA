# Trabajo de inteligencia artificial: clasificación asistida de lesiones cutáneas

## 1. Objetivo y definición del problema

El objetivo principal del proyecto es desarrollar un agente de soporte a la decisión médica para la detección y clasificación lesiones cutáneas a partir de imágenes digitales. 

El problema abordado es la variabilidad en el diagnóstico dermatológico inicial, buscando reducir el riesgo de que lesiones sospechosas pasen desapercibidas. La pidea principal es dar la imagen procesada y etiquetada con el diagnóstico probabilístico de la lesión.

* **Entrada:** Una imagen dermatoscopica.
* **Salida:** La probabilidad de melanoma.


## 2. Origen y naturaleza de los datos

El origen de los datos es del **HAM10000**  (Human Against Machine with 10,000 training images) es un conocido conjunto de datos de imágenes dermatoscópicas utilizado para el entrenamiento y evaluación de modelos de inteligencia artificial en la clasificación de lesiones cutáneas.

### 2.1. Volumen de datos

El dataset cuenta con un total de **10015** imagenes dermatologicas, sin embargo se limitara la cantidad de imagenes usadas, siendo estas solo las referidas a melanomas y lunares benignos.

#### Decisiones

* Se realizo balanceo por submuestreo despues de filtrar, ya que el original contiene muchos mas lunares que melanomas, lo cual puede hacer que el modelo se decante siempre por un lunar, siendo estos falsos negativos
* El test se uso por unica vez para tener metricas honestas.

#### Volumen final: 1228 imagenes con 614 imagenes por clase 



## 3. Métricas de éxito y baseline

### 3.1. Baseline

Modelo de ML clásico (Random Forest) sobre características geométricas y de color extraídas con OpenCV, previa limpieza de pelos y esquinas y segmentación de la lesión con Otsu.

### 3.2. Modelo Propuesto

ResNet18 preentrenada en ImageNet (transfer learning).

### 3.3. Métricas de éxito

* Recall / Sensibilidad: Maximizar la detección de casos de Melanoma apuntando a un 80% de recall,  para reducir al mínimo los falsos negativos en diagnósticos malignos.
* F1-Score macro: Superar de forma objetiva la baseline, buscando alcanzar un F1-Score general mayor al 80% en la clasificación 
* Matriz de Confusión: Evaluar la cantidad de falsos positivos y falsos negativos sobre el conjunto de datos de prueba.

## 4. Límites del alcance

### 4.1. Funciones incluidas

Para esta ocasión nos centraremos principalmente en la deteccion del  melanoma, que este es un tipo de cancer de piel, que puede aparecer como un lunar nuevo o como un lunar existente que cambia, buscando determinar si la lesión detectada es maligna (melanoma) o benigna (lunar común).

### 4.2. Funciones fuera del alcance de la primera entrega

Si bien la funcion principal del sistema busca ayudar con la clasificacion de las diferentes lesiones cutaneas. En este caso solo se tendra en cuenta la deteccion de una lesion en particular.

### 5. Resultados

La evaluación final se realizó sobre un conjunto de prueba balanceado de **246 imágenes**: 123 melanomas y 123 nevos. El umbral de la CNN se eligió previamente con el conjunto de validación y se fijó en **0,7**. Ambos modelos se evaluaron sobre las mismas imágenes:

* **Baseline (Random Forest + OpenCV):**
  * Recall de melanoma: **0,886** (IC 95 %: 0,83–0,94).
  * F1 macro: **0,841** (IC 95 %: 0,79–0,89).
  * Falsos negativos: **14**.
  * Falsos positivos: **25**.
* **CNN (ResNet18):**
  * Recall de melanoma: **0,902** (IC 95 %: 0,85–0,95).
  * F1 macro: **0,886** (IC 95 %: 0,84–0,93).
  * Falsos negativos: **12**.
  * Falsos positivos: **16**.

En esta evaluación, la CNN obtuvo valores más altos de recall y F1 macro, y registró menos falsos negativos y falsos positivos que el baseline. Los intervalos de confianza se estimaron mediante bootstrap con 2.000 remuestreos.

# Instrucciones de uso

## Requisitos

* Python 3.11 (los notebooks se desarrollaron con Python 3.11.9).
* HAM10000: el archivo **HAM10000_metadata.csv** y las imágenes originales deben estar en **data/raw/**  y **data/raw/images/`**, respectivamente.
* Para ejecutar la inferencia se necesita el modelo **models/cnn_resnet18.pt**.

## Preparar el entorno con Docker

Se necesita Docker con Docker Compose. 

Construir la imagen e iniciar Jupyter:

```bash
docker compose up --build
```

Abrir [http://localhost:8888](http://localhost:8888). Si Jupyter solicita un token, usar el enlace con token que aparece en la salida del comando. Para detener el contenedor, presionar `Ctrl+C` y luego ejecutar:

```bash
docker compose down
```

La imagen usa Python 3.11 y las versiones fijadas en `requirements.txt`. Está configurada para CPU y usa `opencv-python-headless`, por lo que no necesita bibliotecas gráficas del sistema. Los datos, modelos y notebooks se montan desde las carpetas del proyecto y no se copian dentro de la imagen.

## Clasificar una imagen

El modelo está diseñado para analizar imágenes dermatoscópicas, similares a las del conjunto HAM10000. No se ha evaluado con fotografías comunes ni con imágenes de otras fuentes. Colocar la imagen de entrada dentro de `imagenesProbar/` y, desde la carpeta raíz, ejecutar:

```bash
docker compose run --rm app python -m src.predecir /app/imagenesProbar/imagen.jpg
```

El programa imprime un resultado JSON con el nombre del archivo, la probabilidad estimada de melanoma, la etiqueta asignada y el umbral utilizado. Para guardar una copia etiquetada en `imagenesProbar/`:

```bash
docker compose run --rm app python -m src.predecir /app/imagenesProbar/imagen.jpg --guardar /app/imagenesProbar/prediccion.jpg
```

La inferencia utiliza el modelo guardado en `models/cnn_resnet18.pt` y el umbral configurado en `src/inferencia.py`. La probabilidad es una salida del modelo y no debe interpretarse como un diagnóstico médico.

## Reproducir el flujo de trabajo

Iniciar Docker con `docker compose up --build`, abrir Jupyter y ejecutar los notebooks en este orden:

1. `01-FiltradoYBalanceoDeDatos.ipynb`: filtra HAM10000, conserva las clases de nevo y melanoma y prepara los datos balanceados en `data/processed/`.
2. `02-SplitEnTrainYTest.ipynb`: crea las particiones de entrenamiento y prueba.
3. `03-ExploracionDeImagenes.ipynb` y `04-Preprocesamiento.ipynb`: explora las imágenes y muestra el preprocesamiento.
4. `05-EstraccionDeCaracteristicas.ipynb`: genera las características para el baseline.
5. `06-BaselineYEvaluacion.ipynb`: entrena y evalúa los modelos baseline.
6. `07-DataSetRed.ipynb`: prepara las particiones de entrenamiento y validación de la CNN.
7. `08-EntrenamientoCNN.ipynb`: entrena la CNN y guarda el modelo en `models/cnn_resnet18.pt`.
8. `09-EvaluacionYComparacion.ipynb`: selecciona el umbral usando validación y compara los modelos sobre el conjunto de prueba.

Para reproducir el entrenamiento, además del contenedor se necesitan los archivos originales de HAM10000 en `data/raw/` y `data/raw/images/`. Ejecutar nuevamente los notebooks de preparación puede regenerar los archivos de `data/processed/`; el notebook 02 también vuelve a crear la partición de prueba.

