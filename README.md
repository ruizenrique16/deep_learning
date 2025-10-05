# Modelos de Deep Learning para Clasificación de Imágenes y Texto

Este repositorio contiene una serie de notebooks que exploran distintos **modelos de Deep Learning** aplicados a problemas de **clasificación**, tanto en imágenes como en texto.  
El objetivo es comparar el desempeño de arquitecturas clásicas (CNN, LSTM) frente a modelos basados en **Transformers** y **fine-tuning** de redes preentrenadas.

-----------------------------------------------------------------------------------------------------

## Estructura del proyecto

├── notebooks/ # Notebooks con los experimentos principales
│ ├── 01_CNN_clasificacion.ipynb
│ ├── 02_LSTM_vs_Transformer.ipynb
│ └── 03_Transformer_Fine_Tuning.ipynb
│
├── results/ # Resultados generados por cada notebook (CSV, métricas, gráficas)
│
└── docs/ # Reportes técnicos en formato Markdown (.md)


-----------------------------------------------------------------------------------------------------

## Contenido de los experimentos

### 1. Clasificación de imágenes con CNN
- **Dataset:** CIFAR-10  
- **Técnicas:** normalización, aumento de datos, regularización L2, *EarlyStopping*.  
- **Modelo:** Red Convolucional profunda (*CNN*) para clasificación en 10 categorías.  
- **Salida:** precisión, pérdida, matriz de confusión y reporte de clasificación.  

Reporte técnico: `docs/01_CNN_clasificacion.md`  
Resultados:      `results/cnn_results.csv`

-----------------------------------------------------------------------------------------------------

### 2. Modelos secuenciales — LSTM vs Transformer
- **Dataset:** IMDb (reseñas de películas).  
- **Técnicas:** tokenización, *padding*, embeddings, comparación de arquitecturas secuenciales.  
- **Modelos:** LSTM y Transformer implementados con Keras/TensorFlow.  
- **Salida:** métricas de evaluación, matrices de confusión y resultados comparativos.  

Reporte técnico: `docs/02_LSTM_vs_Transformer.md`  
Resultados: `results/lstm_transformer_results.csv`

-----------------------------------------------------------------------------------------------------

### 3. Fine-tuning de Transformer preentrenado (DistilBERT)
- **Dataset:** IMDb.  
- **Modelo:** `distilbert-base-uncased` (Hugging Face Transformers).  
- **Técnicas:** ajuste fino (*fine-tuning*) y evaluación con `Trainer` de 🤗 Transformers.  
- **Comparación:** desempeño frente a LSTM y Transformer del experimento anterior.  

Reporte técnico: `docs/03_Transformer_Fine_Tuning.md`  
Resultados: `results/fine_tuning_results.csv`

-----------------------------------------------------------------------------------------------------

## Requisitos

Para ejecutar las notebooks se recomienda crear un entorno con las siguientes librerías:

```bash
pip install tensorflow torch transformers datasets evaluate scikit-learn seaborn matplotlib pandas numpy
```

Se sugiere usar Python 3.10+ y ejecutar las notebooks desde Jupyter o Google Colab.

## Ejecución

Abrir la carpeta notebooks/

Ejecutar cada notebook en orden:

01_CNN_clasificacion.ipynb

02_LSTM_vs_Transformer.ipynb

03_Transformer_Fine_Tuning.ipynb

✒️ Autor
Pedro Enrique Ruiz Riveros
📧 Contacto: [enriq16@fpuna.edu.py]
📚 Portafolio de proyectos de Deep Learning y NLP