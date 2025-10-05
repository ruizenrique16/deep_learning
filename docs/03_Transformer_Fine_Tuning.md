# Proyecto 03 – Fine-tuning de Transformer (DistilBERT) para clasificación de sentimiento IMDb  
**Autor:** Pedro Enrique Ruiz Riveros
**Fecha:** 6 de octubre de 2025  

--------------------------------------------------------------------------------------------

## Resumen ejecutivo
En esta tercera entrega entrenamos **DistilBERT** con el dataset IMDb para decidir si una reseña de cine es positiva o negativa.  
A diferencia de los modelos anteriores (LSTM y Transformer desde-cero), aquí partimos de **pesos pre-entrenados en inglés** y solo ajustamos la cabeza de clasificación.  
Con **2 épocas y 25 000 reseñas** alcanzamos **89,2 % de accuracy** en apenas **7 minutos** de entrenamiento.

--------------------------------------------------------------------------------------------

## 1. Problema y datasets
**Tarea:** Clasificación binaria de sentimiento (positivo / negativo)  
**Dataset:** [IMDb Movie Reviews](https://huggingface.co/datasets/imdb)  
- 25 000 reseñas para entrenamiento  
- 25 000 para test (para el test utilizamos 5000 muestras) 


**Pre-proceso:**  
- Tokenización con **WordPiece** (`distilbert-base-uncased`)  
- Padding / truncamiento a 512 tokens  
 

--------------------------------------------------------------------------------------------

## 2. Metodología – Módulo Fine-Tuning

### 2.1 Arquitectura
La arquitectura parte del modelo DistilBERT, una versión reducida de BERT que conserva 6 capas de transformadores y 66 millones de parámetros. Tras procesar la secuencia de tokens, se extrae el vector asociado al token [CLS] —que resume el contenido de toda la oración—, se aplica un dropout del 10 % para regularizar, y finalmente se pasa por una capa densa de 2 neuronas con activación sigmoide que produce la probabilidad de pertenecer a la clase positiva o negativa.


### 2.2 Hiper-parámetros clave
| Parámetro     | Valor                        |
|---------------|---------------------------   |
| Modelo base   | `distilbert-base-uncased`    |
| Max length    | 512 tokens                   |
| Batch size    | 16 (TPU v5e)                 |
| Learning rate | 2 × 10⁻⁵                     |
| Epochs        | 2                            |
| Optimizador   | AdamW (`optim="adamw_torch"`)|
| Weight decay  | 0.01                         |
| Dropout       | 0.1 (interno)                |

### 2.3 Recursos computacionales
- **Entorno:** Google Colab TPU v5e  
- **Tiempo total:** 6 min 54 s (414 s)  
- **Memoria:** ≈ 8 GB RAM  
- **Parámetros entrenables:** 66 955 010 (solo cabeza + última capa de BERT)

---

## 3. Resultados
| Métrica                 | Valor   |
|-------------------------|---------|
| Accuracy                | 89.24 % |
| F1-macro                | 89.24 % |
| Loss test               | 0.315   |
| Épocas corridas         |    2    |
| Tiempo entrenamiento    | 414 s   |
| Tiempo inferencia (5 k) | 10.8 s  |

