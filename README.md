## Descripción de la Red Neuronal
### La red neuronal está compuesta por:

- Capa de Entrada: 3 características (variables de entrada) de las listas de reproducción en Spotify, Apple Music y Deezer.
- Capa Oculta: 1 capa con 8 neuronas y función de activación ReLU.
- Capa de Salida: 1 neurona con función de activación sigmoide para clasificar si la canción supera o no los 100 millones de streams.
- Preprocesamiento de los Datos:
  - Convertir a Valores Numéricos: Convierte todas las columnas a datos numéricos.
  - Eliminar NaN: Se eliminan las columnas y filas que contienen valores NaN.
  - Normalizar Datos: Normaliza las características de entrada para mejorar el rendimiento de la red.
  - Eliminar Outliers: Usa el percentil para eliminar valores atípicos, ya que los datos están fuertemente sesgados a la derecha.
Entrenamiento
  Para entrenar la red neuronal, ejecuta el script principal train.py:


## Parámetros de Entrenamiento
- Tasa de Aprendizaje: learning_rate=0.01
- Regularización L2: lambda_l2=0.01
- Dropout: dropout=0.2
- Número de Épocas: 5000 

## Como usar la red

 - Los requerimientos de este proyecto son:
     - python 3.12
     - pandas 
     - numpy 
     - matplotlib
     - seaborn 
     - scikit-learn

### Para correr el proyecto es necesario ejecutar el archivo `run.py`
