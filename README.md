# Trabajo de inteligencia artificial: clasificación asistida de lesiones cutáneas

## 1. Objetivo y definición del problema

El objetivo principal del proyecto es desarrollar un agente de soporte a la decisión médica para la detección y clasificación lesiones cutáneas a partir de imágenes digitales. 

El problema abordado es la variabilidad en el diagnóstico dermatológico inicial, buscando reducir el riesgo de que lesiones sospechosas pasen desapercibidas. La pidea principal es dar la imagen procesada y etiquetada con el diagnóstico probabilístico de la lesión.

## 2. Origen y naturaleza de los datos

El origen de los datos es del **HAM10000**  (Human Against Machine with 10,000 training images) es un conocido conjunto de datos de imágenes dermatoscópicas utilizado para el entrenamiento y evaluación de modelos de inteligencia artificial en la clasificación de lesiones cutáneas.

### 2.1. Volumen de datos

El dataset cuenta con un total de **10015** imagenes dermatologicas, sin embargo se limitara la cantidad de imagenes usadas, siendo estas solo las referidas a melanomas y lunares benignos.

## 3. Métricas de éxito y baseline

### 3.1. Baseline

Para evaluar si aporta una mejora real frente a un método tradicional, se utilizará como baseline un modelo de aprendizaje automático simple  entrenado sobre características geométricas y de color extraídas directamente con **OpenCV**

### 3.2. Métricas de éxito

* Recall / Sensibilidad: Maximizar la detección de casos de Melanoma apuntando a un 80% de recall,  para reducir al mínimo los falsos negativos en diagnósticos malignos.
* F1-Score macro: Superar de forma objetiva la baseline, buscando alcanzar un F1-Score general mayor al 80% en la clasificación 
* Matriz de Confusión: Evaluar la cantidad de falsos positivos y falsos negativos sobre el conjunto de datos de prueba.

## 4. Límites del alcance

### 4.1. Funciones incluidas

Para esta ocasión nos centraremos principalmente en la deteccion del  melanoma, que este es un tipo de cancer de piel, que puede aparecer como un lunar nuevo o como un lunar existente que cambia, buscando determinar si la lesión detectada es maligna (melanoma) o benigna (lunar común).

### 4.2. Funciones fuera del alcance de la primera entrega

Si bien la funcion principal del sistema busca ayudar con la clasificacion de las diferentes lesiones cutaneas. En este caso solo se tendra en cuenta la deteccion de una lesion en particular.