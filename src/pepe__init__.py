# project_root/
# │
# ├── data/
# │   ├── raw/                   # Datos originales sin procesar
# │   ├── processed/             # Datos preprocesados listos para el modelo
# │   └── external/              # Datos adicionales o externos (e.g., preentrenados)
# │
# ├── notebooks/                 # Notebooks Jupyter para experimentación y EDA
# │   ├── eda.ipynb              # Exploratory Data Analysis (EDA)
# │   └── experiments.ipynb      # Notebooks de pruebas y prototipos
# │
# ├── src/                       # Código fuente
# │   ├── pepe__init__.py
# │   ├── data/
# │   │   ├── data_loader.py     # Carga y procesamiento de datos
# │   │   └── data_transformer.py# Transformaciones y preprocesamiento
# │   ├── models/
# │   │   ├── model.py           # Definición de la arquitectura de la red neuronal
# │   │   ├── trainer.py         # Funciones para entrenamiento
# │   │   └── evaluator.py       # Evaluación de modelos (métricas y validación)
# │   ├── utils/
# │   │   ├── helpers.py         # Funciones de utilidad general
# │   │   └── visualization.py   # Visualización de datos y resultados
# │   └── config.py              # Configuración del proyecto (hiperparámetros, rutas)
# │
# ├── tests/                     # Pruebas unitarias y de integración
# │   ├── test_data_loader.py
# │   ├── test_model.py
# │   └── ...
# │
# ├── logs/                      # Logs de entrenamiento y resultados
# │
# ├── saved_models/              # Modelos entrenados y checkpoints
# │
# ├── requirements.txt           # Dependencias y paquetes necesarios
# ├── README.md                  # Descripción del proyecto y cómo usarlo
# └── run.py                     # Script principal para ejecutar el entrenamiento
