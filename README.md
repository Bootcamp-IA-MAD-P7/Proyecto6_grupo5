# Proyecto6_grupo5

**Clasificación Multiclase de Semillas de Judías Utilizando Modelos Ensemble y MLOps**

---

## Descripción del Proyecto

Este proyecto clasifica variedades de semillas de judías (*Dry Bean*) a partir de características extraídas de imágenes, utilizando **modelos de ensamble** y siguiendo buenas prácticas de **MLOps** (tracking con MLflow, despliegue con Docker, tests automatizados).

El dataset utilizado es el **UCI Dry Bean Dataset**, que contiene 13,611 muestras de 7 clases de judías: `SEKER`, `BARBUNYA`, `BOMBAY`, `CALI`, `HOROZ`, `SIRA` y `DERMASON`. Tras la limpieza (eliminación de duplicados), el dataset de trabajo tiene **13,543 muestras** con 16 características numéricas.

---

## Estado Actual

| Fase | Estado |
|------|--------|
| Estructura del repositorio | Creada |
| Dataset descargado y limpiado | `data/raw/` + `data/processed/` |
| EDA (Análisis Exploratorio) | Completado — `notebooks/01_eda.ipynb` |
| Preprocesamiento + Modelado base | Completado — `notebooks/02_preprocesamiento_modelado.ipynb` |
| Modelos ensemble avanzados | Completado — `notebooks/03_modelos_ensemble.ipynb` |
| Optimización de hiperparámetros | Completado — `notebooks/04_optimizacion_hiperparametros.ipynb` |
| Pipeline modular (src/) | Implementado — `src/preprocessing.py`, `src/train.py`, `src/predict.py` |
| App Streamlit (EDA + Predicción) | Implementada — `app/main.py` (6 pestañas) |
| Tests unitarios | Implementados — 15 tests, todos pasando |
| MLflow tracking | Implementado — `src/run_pipeline.py` |
| Despliegue Docker | Implementado — `Dockerfile` + `docker-compose.yml` |

---

## Resultados de los Modelos

### Modelos base (notebook 02)

| Modelo | Accuracy Train | Accuracy Test | Diferencia |
|--------|---------------|---------------|------------|
| Regresión Logística | 92.6% | 91.9% | 0.68% |
| Random Forest | 94.2% | 91.5% | 2.77% |

### Modelos ensemble avanzados (notebook 03)

Se comparan Gradient Boosting, XGBoost y LightGBM frente a los modelos base. Ejecutar el notebook para ver la tabla completa.

### Optimización de hiperparámetros (notebook 04)

Búsqueda aleatoria (`RandomizedSearchCV`, 50 iteraciones, 5-fold CV) sobre Random Forest y Gradient Boosting.

---

## Estructura del Repositorio

```
.
├── app/
│   └── main.py                          # App Streamlit (EDA Explorer + Predicción)
├── data/
│   ├── raw/
│   │   ├── Dry_Bean.csv
│   │   └── Dry_Bean_Dataset.xlsx
│   └── processed/
│       └── dry_bean_clean.csv
├── models/
│   ├── best_model.pkl                   # Mejor modelo serializado
│   ├── random_forest_model.pkl          # Random Forest base
│   ├── scaler.pkl                       # StandardScaler
│   ├── label_encoder.pkl                # LabelEncoder
│   ├── comparison_chart.png             # Gráfico comparativo
│   ├── confusion_matrix_best.png        # Matriz de confusión
│   └── feature_importance_best.png      # Importancia de features
├── notebooks/
│   ├── 01_eda.ipynb                     # Análisis exploratorio
│   ├── 02_preprocesamiento_modelado.ipynb  # Modelado base
│   ├── 03_modelos_ensemble.ipynb        # Ensemble avanzados
│   └── 04_optimizacion_hiperparametros.ipynb  # Hiperparámetros
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                 # Carga, limpieza, split de datos
│   ├── train.py                         # Entrenamiento y evaluación
│   ├── predict.py                       # Pipeline de predicción
│   └── run_pipeline.py                  # Pipeline completo con MLflow
├── tests/
│   └── test_model.py                    # 15 tests unitarios
├── .streamlit/config.toml               # Configuración tema Streamlit
├── .dockerignore
├── Dockerfile
├── docker-compose.yml
├── Instructivo.md
└── requirements.txt
```

---

## Requisitos Previos

- **Python** >= 3.12
- **pip** o **uv** — gestor de paquetes
- **Docker** (opcional, para despliegue containerizado)

---

## Instalación

```powershell
# 1. Clonar el repositorio
git clone https://github.com/Bootcamp-IA-MAD-P7/Proyecto6_grupo5
cd Proyecto6_grupo5

# 2. Crear entorno virtual
python -m venv .venv

# 3. Activar el entorno
# PowerShell:
.venv\Scripts\Activate.ps1
# Bash / Linux / macOS:
source .venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt
```

---

## Cómo Ejecutar

### App Streamlit (EDA Explorer + Predicción)

```bash
streamlit run app/main.py
```

Se abre en `http://localhost:8501`. Incluye 6 pestañas:
- **Distribución**: barras de clases, histogramas y boxplots
- **Relaciones**: scatter plots y violin plots
- **Correlación**: heatmap con Pearson/Spearman/Kendall
- **Estadísticas**: tabla descriptiva por clase
- **Datos**: explorador con búsqueda y descarga CSV
- **Predicción**: formulario de 16 features con predicción del modelo y probabilidades

### Docker

```bash
# Construir y ejecutar
docker compose up --build

# O solo el build
docker build -t bean-classifier .
docker run -p 8501:8501 bean-classifier
```

### Pipeline con MLflow

```bash
python src/run_pipeline.py
```

Entrena 3 modelos, registra métricas y hiperparámetros en MLflow, y guarda el mejor modelo. Para ver la UI de MLflow:

```bash
mlflow ui
```

### Tests

```bash
python -m pytest tests/ -v
```

15 tests cubriendo: existencia de dataset, columnas esperadas, valores nulos, clases, artefactos del modelo, forma del scaler, predicciones, confianza y límite de memoria.

### Notebooks

```bash
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_preprocesamiento_modelado.ipynb
jupyter notebook notebooks/03_modelos_ensemble.ipynb
jupyter notebook notebooks/04_optimizacion_hiperparametros.ipynb
```

---

## Módulos de src/

| Módulo | Función principal |
|--------|-------------------|
| `preprocessing.py` | `load_raw_data()`, `clean_data()`, `get_feature_target_split()`, `scale_features()` |
| `train.py` | `get_models()`, `train_all_models()`, `results_to_dataframe()`, `save_model()` |
| `predict.py` | `load_artifacts()`, `predict_single()`, `predict_batch()` |
| `run_pipeline.py` | Pipeline completo con logging a MLflow |

---

## Clases del Dataset

| Clase | Muestras | Descripción |
|-------|----------|-------------|
| `DERMASON` | 3,546 | Judía con forma de media luna |
| `SIRA` | 2,636 | Judía intermedia |
| `SEKER` | 2,027 | Judía redonda pequeña |
| `HOROZ` | 1,860 | Judía alargada y delgada |
| `CALI` | 1,630 | Judía grande y ancha |
| `BARBUNYA` | 1,322 | Judía de forma irregular |
| `BOMBAY` | 522 | Judía grande y alargada |

---

## Dependencias

| Paquete | Versión mínima | Uso |
|---------|----------------|-----|
| streamlit | >= 1.35.0 | App web interactiva |
| pandas | >= 2.2.0 | Manipulación de datos |
| numpy | >= 1.26.0 | Computación numérica |
| plotly | >= 5.22.0 | Gráficos interactivos |
| matplotlib | >= 3.9.0 | Gráficos estáticos |
| seaborn | >= 0.13.0 | Visualización estadística |
| scikit-learn | >= 1.4.0 | Modelos y preprocesamiento |
| joblib | >= 1.3.0 | Serialización de modelos |
| xgboost | >= 2.0.0 | Modelo ensemble XGBoost |
| lightgbm | >= 4.0.0 | Modelo ensemble LightGBM |
| mlflow | >= 2.10.0 | Tracking de experimentos |
| pytest | >= 8.0.0 | Tests unitarios |

---

## Flujo de trabajo con Git

1. Nunca trabajar directo en `main` ni en `develop`.
2. Crear tu rama: `git checkout -b feature/nombre-de-tu-tarea`
3. Antes de empezar: `git checkout develop && git pull && git checkout feature/tu-rama && git merge develop`
4. Al terminar: commit + push, y abrir PR hacia `develop`.

---

## Referencias

- [UCI Machine Learning Repository — Dry Bean Dataset](https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
