# Proyecto6_grupo5

**Clasificación Multiclase de Semillas de Judías Utilizando Modelos Ensemble y MLOps**

---

## Descripción del Proyecto

Este proyecto tiene como objetivo clasificar variedades de semillas de judías (*Dry Bean*) a partir de características extraídas de imágenes, utilizando **modelos de ensamble** (Random Forest, Gradient Boosting, etc.) y siguiendo buenas prácticas de **MLOps** para la gestión del ciclo de vida de los modelos.

El dataset utilizado es el **UCI Dry Bean Dataset**, que contiene 13,611 muestras de 7 clases de judías: `SEKER`, `BARBUNYA`, `BOMBAY`, `CALI`, `HOROZ`, `SIRA` y `DERMASON`. Cada muestra incluye 16 características numéricas calculadas a partir de imágenes (área, perímetro, eje mayor, eje menor, excentricidad, etc.).

---

## Estado Actual

| Fase | Estado |
|------|--------|
| Estructura del repositorio | Creada |
| Dataset descargado | Disponible en `data/raw/Dry_Bean.csv` |
| EDA (Análisis Exploratorio de Datos) | En progreso (`notebooks/01_eda.ipynb`) |
| Preprocesamiento / Feature Engineering | Pendiente |
| Entrenamiento de modelos | Pendiente |
| Evaluación de modelos | Pendiente |
| MLOps (pipeline, tracking, despliegue) | Pendiente |
| Tests | Pendientes (`tests/` vacío) |
| App (API / interfaz) | Pendiente (`app/` vacío) |

---

## Estructura del Repositorio

```
.
├── app/                    # Aplicación (API, interfaz, despliegue)
├── data/
│   ├── raw/                # Dataset original
│   │   └── Dry_Bean.csv
│   └── processed/          # Datos preprocesados (generados durante el pipeline)
├── models/                 # Modelos entrenados serializados
├── notebooks/
│   └── 01_eda.ipynb        # Análisis exploratorio de datos
├── src/                    # Código fuente del pipeline (preprocesamiento, modelos, utils)
├── tests/                  # Tests unitarios y de integración
├── Instructivo.md          # Guía de instalación y configuración
└── requirements.txt        # Dependencias del proyecto
```

---

## Requisitos Previos

- **Python** >= 3.12
- **uv** — gestor de entornos y paquetes (instalación: ver `Instructivo.md`)
- Opcionalmente: **Docker** (si se usa para despliegue)

---

## Instalación

```powershell
# 1. Clonar el repositorio
git clone <url-del-repositorio>
cd Proyecto6_grupo5

# 2. Crear entorno virtual con uv
uv venv

# 3. Activar el entorno
# PowerShell:
.venv\Scripts\Activate.ps1
# Bash / Linux / macOS:
source .venv/bin/activate

# 4. Instalar dependencias
uv pip install -r requirements.txt
```

> **Nota:** El `requirements.txt` actual es una lista parcial. Se recomienda actualizarlo con las dependencias completas del proyecto (matplotlib, numpy, pandas, seaborn, scikit-learn, etc.) a medida que se avanza.

---

## Cómo Ejecutar

### Análisis Exploratorio (EDA)

```bash
# Abrir con Jupyter
jupyter notebook notebooks/01_eda.ipynb

# O desde VS Code, abrir el notebook directamente
```

### Pipeline de Entrenamiento

> Pendiente — se implementará en `src/` con scripts para preprocesamiento, entrenamiento y evaluación.

---

## Clases del Dataset

| Clase | Descripción |
|-------|-------------|
| `SEKER` | Judía redonda pequeña |
| `BARBUNYA` | Judía de forma irregular |
| `BOMBAY` | Judía grande y alargada |
| `CALI` | Judía grande y ancha |
| `HOROZ` | Judía alargada y delgada |
| `SIRA` | Judía intermedia |
| `DERMASON` | Judía con forma de media luna |

---

## Notas para el Desarrollo

- El notebook `01_eda.ipynb` incluye ya: distribución de clases, boxplots de área por clase, matriz de correlación y dispersión área vs. excentricidad.
- La columna `Class` del dataset tiene valores con espacios en blanco al final; se limpia con `.str.strip()` en el EDA.
- Las imágenes generadas por los gráficos se embeben en el notebook (formato base64). Al ejecutar localmente, las gráficas se muestrarán directamente.
- Se recomienda seguir la convención de nombrar los notebooks secuencialmente: `01_eda.ipynb`, `02_preprocesamiento.ipynb`, `03_modelos.ipynb`, etc.

---

## Referencias

- [UCI Machine Learning Repository — Dry Bean Dataset](https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset)
