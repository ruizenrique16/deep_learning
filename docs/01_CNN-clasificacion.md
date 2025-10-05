# Reporte Técnico: Clasificación de Imágenes con una CNN

**Autor:** Pedro Enrique Ruiz Riveros  
**Fecha:** Octubre 2025  
**Archivo fuente:** `01_cnn_clasificacion.py`

---

## Resumen Ejecutivo
Este proyecto implementa una red neuronal convolucional (CNN) para la clasificación de imágenes del conjunto de datos CIFAR-10. Se entrenó un modelo profundo con cuatro capas convolucionales, capas de pooling y una capa totalmente conectada con regularización L2, utilizando aumento de datos (data augmentation) para mejorar la generalización.

---

## Problema y Dataset
El objetivo es clasificar imágenes en **10 categorías**: avión, automóvil, pájaro, gato, venado, perro, rana, caballo, barco y camión.  
Se utilizó el dataset **CIFAR-10**, que contiene **60.000 imágenes a color de 32×32 píxeles**, distribuidas en:
- 50.000 para entrenamiento  
- 10.000 para prueba  

Las imágenes fueron **normalizadas** a un rango de valores entre 0 y 1.

---

## Metodología

### Arquitectura del Modelo

Se configuro una arquitectura con 4 capas convolucionales seguidas de eguidas de capas de pooling para la reducción espacial de las características, y finalizando con capas densas (fully connected) para la clasificación final.

Cada capa convolucional utiliza la función de activación ReLU y aplica regularización L2 para evitar el sobreajuste. Las primeras capas detectan patrones simples como bordes y texturas, mientras que las capas más profundas capturan características más abstractas.

Después de las capas convolucionales y de pooling, la salida se aplana (flatten) y se conecta a una capa densa intermedia de 256 neuronas, también con ReLU y regularización L2. Finalmente, la capa de salida utiliza softmax para obtener las probabilidades de pertenencia a cada una de las 10 clases del dataset CIFAR-10.

### Hiperparámetros Clave
- Optimizador: **Adam**  
- Tasa de aprendizaje: **0.001**  
- Regularización L2: **1e-4**  
- Tamaño de batch: **128**  
- Épocas: hasta **80** con *EarlyStopping* (paciencia = 8)  
- **Data augmentation:**
  - Rotación: ±15°  
  - Desplazamiento: 10%  
  - Zoom: 10%  
  - Volteo horizontal  

### Recursos Computacionales
- GPU de Google Colab (NVIDIA T4)  
- Librerías: TensorFlow 2.x, Keras, NumPy, Matplotlib, Seaborn, scikit-learn

---

## Resultados y Discusión

### Curvas de Entrenamiento
Las curvas de pérdida y exactitud muestran una convergencia estable, con una diferencia moderada entre los conjuntos de entrenamiento y validación, lo que indica un buen nivel de generalización gracias a la regularización y al aumento de datos.

### Exactitud del Modelo

| Conjunto      | Exactitud         | Pérdida  |
|---------------|-------------------|----------|
| Entrenamiento | ~0.85             | ~0.62    |
| Prueba        | **≈ 0.80–0.83**   | ~0.70    |

*(Los valores exactos dependen del run, ya que se usa data augmentation).*

### Matriz de Confusión
Se observa una buena clasificación general en la mayoría de las clases. Los errores más comunes ocurren entre categorías visualmente similares (por ejemplo, **gato ↔ perro**, **automóvil ↔ camión**).

### Informe de Clasificación
El *classification report* muestra valores **F1-score > 0.80** en la mayoría de las clases, indicando un desempeño equilibrado entre precisión y exhaustividad.

### Discusión
- **Lo que funcionó:**  
  - El uso de **regularización L2** y **data augmentation** redujo el sobreajuste.  
  - El optimizador **Adam** aceleró la convergencia.  
  - *EarlyStopping* evitó el entrenamiento innecesario.  

- **Lo que no funcionó tan bien:**  
  - Las clases con similitud visual presentaron confusiones.  
  - Se podría mejorar la precisión agregando *Batch Normalization* o una arquitectura más profunda (como ResNet o VGG-like).

---
