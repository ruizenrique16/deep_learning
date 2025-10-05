# Proyecto 2 – Modelos Secuenciales: LSTM vs. Transformer  
**Autor:** *Pedro Enrique Ruiz Riveros*  
**Fecha:** 5 de octubre de 2025  

------------------------------------------------------------------

## Resumen ejecutivo
En este trabajo comparamos dos arquitecturas de deep-learning para clasificación de sentimiento en reseñas de cine (IMDb):  
1. **LSTM doble**: capas recurrentes que leen la secuencia palabra a palabra y retienen contexto a través de estados ocultos.  
2. **Transformer ligero**: bloques de auto-atención que ponderan globalmente cada token, sin recurrencia.  

Ambos modelos alcanzan ~89-90 % de accuracy en test, pero el Transformer entrena 30 % más rápido y posee 25 % menos parámetros. El código, notebooks y métricas están liberados bajo MIT para que puedas replicar o citar en entrevistas.

------------------------------------------------------------------

## 1. Problema y datasets
**Tarea:** Clasificación binaria de sentimiento (positivo / negativo)  
**Dataset:** [Large Movie Review – IMDb](https://ai.stanford.edu/~amaas/data/sentiment/)  
- 50 000 reseñas (25 k train + 25 k test) balanceadas.  
- Longitud media ≈ 230 palabras; maxlen fijado en 200 tokens.  
**Pre-proceso:**  
- Tokenizer Keras sobre vocabulario de 20 000 palabras más frecuentes.  
- Padding / truncamiento post para homogeneizar longitud.  

------------------------------------------------------------------

## 2. Metodología por módulo

### 2.1 LSTM
**Arquitectura:**  
Este modelo está basado en redes recurrentes (RNNs) con celdas LSTM, que son capaces de recordar información a largo plazo.
**Estructura:**
- Embedding: convierte palabras en vectores densos de 128 dimensiones.
- Primera LSTM: 128 unidades, con return_sequences=True para pasar toda la secuencia a la siguiente capa.
- Segunda LSTM: 64 unidades, que solo devuelve el último estado (resumen de la secuencia).
- Capas densas: una capa intermedia de 64 neuronas con activación ReLU, seguida de dropout para evitar sobreajuste.
- Salida: una neurona con activación sigmoide para clasificación binaria (positivo/negativo).

**Hiper-parámetros clave:**  
- batch_size = 64  
- early-stopping (patience = 3, monitor = val_loss)  
- optimizer = Adam (lr por defecto)  
**Recursos:**  
- GPU Tesla T4 (Google Colab)  
- Tiempo de entrenamiento ≈ 7 min (≈ 4 épocas hasta early-stop)  
- Parámetros totales ≈ 2.75 M  

### 2.2 Transformer mini
**Arquitectura:** 
Este modelo utiliza el mecanismo de auto-atención en lugar de recurrencia, lo que permite procesar toda la secuencia en paralelo.

**Estructura:**
- Embedding: convierte palabras en vectores de 64 dimensiones.
- Bloques Transformer: 2 bloques con auto-atención multi-cabeza (4 cabezas) y redes feedforward (128 unidades). Cada bloque incluye normalización y conexiones residuales.
- GlobalAveragePooling1D: resume toda la secuencia en un solo vector.
- Capas densas: una capa de 64 neuronas con ReLU y dropout para regularización.
- Salida: una neurona con activación sigmoide para clasificación binaria.
**Hiper-parámetros clave:**  
- d_model = 64 (reduce memoria vs. 128)  
- rate_dropout = 0.1 dentro de bloques, 0.4/0.3 en cabezales  
- batch_size = 64, early-stopping idéntico  
**Recursos:**  
- Misma GPU; tiempo de entrenamiento ≈ 5 min (≈ 3 épocas)  
- Parámetros totales ≈ 2.1 M  

------------------------------------------------------------------

## 3. Resultados resumidos
| Modelo     | Acc. Test | F1-macro | Épocas | Params   | t_train |
|------------|-----------|----------|--------|----------|---------|
| LSTM       | 0.890     | 0.889    | 4      | 2.75 M   | 420 s   |
| Transformer| 0.893     | 0.892    | 3      | 2.10 M   | 310 s   |

