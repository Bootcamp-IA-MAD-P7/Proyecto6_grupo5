# Proyecto6_grupo5

**Clasificación Multiclase de Semillas de Judías Utilizando Modelos Ensemble y MLOps**

---

## Descripción del Proyecto

Este proyecto tiene como objetivo clasificar variedades de semillas de judías (_Dry Bean_) a partir de características extraídas de imágenes, utilizando **modelos de ensamble** (Random Forest, y próximamente Gradient Boosting/AdaBoost) y siguiendo buenas prácticas de **MLOps** para la gestión del ciclo de vida de los modelos.

El dataset utilizado es el **UCI Dry Bean Dataset**, que contiene 13,611 muestras de 7 clases de judías: `SEKER`, `BARBUNYA`, `BOMBAY`, `CALI`, `HOROZ`, `SIRA` y `DERMASON`. Cada muestra incluye 16 características numéricas calculadas a partir de imágenes (área, perímetro, eje mayor, eje menor, excentricidad, etc.).

---

## Estado Actual

| Fase                                   | Estado                                                                               |
| -------------------------------------- | ------------------------------------------------------------------------------------ |
| Estructura del repositorio             | ✅ Completo                                                                          |
| Dataset descargado                     | ✅ Disponible en `data/raw/Dry_Bean.csv`                                             |
| EDA (Análisis Exploratorio de Datos)   | ✅ Completo (`notebooks/01_eda.ipynb`)                                               |
| Preprocesamiento / Feature Engineering | ✅ Completo (`notebooks/02_preprocesamiento_modelado.ipynb`)                         |
| Entrenamiento de modelos               | ✅ Completo (Regresión Logística + Random Forest)                                    |
| Evaluación de modelos                  | ✅ Completo (accuracy, precision/recall/F1, matriz de confusión, feature importance) |
| MLOps (pipeline, tracking, despliegue) | Pendiente                                                                            |
| Tests                                  | Pendiente (`tests/` vacío)                                                           |
| App (dashboard EDA + predicción)       | 🔶 En progreso — dashboard EDA listo, falta página de predicción                     |

---

## Resultados del Modelo

| Modelo                         | Accuracy Train | Accuracy Test | Overfitting |
| ------------------------------ | -------------- | ------------- | ----------- |
| Regresión Logística (baseline) | 92.60%         | 91.92%        | 0.68%       |
| **Random Forest (ensemble)**   | 94.24%         | 91.47%        | **2.77%**   |

✅ Overfitting controlado en ambos modelos, por debajo del 5% requerido.

Artefactos guardados en `models/`:

- `random_forest_model.pkl` — modelo final
- `scaler.pkl` — escalador de features
- `label_encoder.pkl` — codificador de las 7 clases

---

## Estructura del Repositorio

├── app/ # Aplicación Streamlit (dashboard EDA + predicción)
│ └── main.py
├── data/
│ ├── raw/ # Dataset original
│ │ └── Dry_Bean.csv
│ └── processed/ # Dataset limpio y procesado
│ └── dry_bean_clean.csv
├── models/ # Modelos entrenados serializados
│ ├── random_forest_model.pkl
│ ├── scaler.pkl
│ └── label_encoder.pkl
├── notebooks/
│ ├── 01_eda.ipynb # Análisis exploratorio de datos
│ └── 02_preprocesamiento_modelado.ipynb # Preprocesamiento, modelado y evaluación
├── src/ # Código fuente del pipeline
├── tests/ # Tests unitarios y de integración
├── Instructivo.md # Guía de instalación y configuración
└── requirements.txt # Dependencias del proyecto

---

## Requisitos Previos

- **Python** >= 3.12
- Entorno virtual (`venv`) o **uv** — ver `Instructivo.md`

---

## Instalación

```powershell
# 1. Clonar el repositorio
git clone https://github.com/Bootcamp-IA-MAD-P7/Proyecto6_grupo5.git
cd Proyecto6_grupo5

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar el entorno
# Git Bash / Windows:
source venv/Scripts/activate
# PowerShell:
venv\Scripts\Activate.ps1

# 4. Instalar dependencias
pip install -r requirements.txt
```

---

## Cómo Ejecutar

### Análisis Exploratorio (EDA)

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Preprocesamiento y Modelado

```bash
jupyter notebook notebooks/02_preprocesamiento_modelado.ipynb
```

Este notebook genera los artefactos en `models/` (modelo, scaler, label encoder) necesarios para la app.

### App (Streamlit)

```bash
streamlit run app/main.py
```

---

## Clases del Dataset

| Clase      | Descripción                   |
| ---------- | ----------------------------- |
| `SEKER`    | Judía redonda pequeña         |
| `BARBUNYA` | Judía de forma irregular      |
| `BOMBAY`   | Judía grande y alargada       |
| `CALI`     | Judía grande y ancha          |
| `HOROZ`    | Judía alargada y delgada      |
| `SIRA`     | Judía intermedia              |
| `DERMASON` | Judía con forma de media luna |

---

## Notas para el Desarrollo

- El notebook `01_eda.ipynb` incluye: distribución de clases, boxplots de área por clase, matriz de correlación y dispersión área vs. excentricidad.
- El notebook `02_preprocesamiento_modelado.ipynb` incluye: encoding, escalado, split estratificado, modelo baseline, modelo ensemble, validación cruzada estratificada, métricas completas y guardado de artefactos.
- La columna `Class` del dataset tiene valores con espacios en blanco al final; se limpia con `.str.strip()` en el EDA.
- Se usa `Develop` (con mayúscula) como rama de integración del equipo.

---

## Referencias

- [UCI Machine Learning Repository — Dry Bean Dataset](https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset)
