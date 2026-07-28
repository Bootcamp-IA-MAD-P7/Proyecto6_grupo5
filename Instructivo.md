# Instructivo del Proyecto — Dry Bean Classification (Grupo 5)

## Requisitos previos

- Python 3.12.10 instalado
- Git instalado
- VSCode con la extensión de Jupyter

## Configuración del entorno (primera vez)

1. Clona el repositorio y entra en la carpeta:
   git clone https://github.com/Bootcamp-IA-MAD-P7/Proyecto6_grupo5
   cd Proyecto6_grupo5

2. Crea el entorno virtual:
   python -m venv venv

3. Actívalo:
   - Windows (Git Bash): source venv/Scripts/activate
   - Windows (CMD): venv\Scripts\activate.bat

4. Instala las dependencias:
   pip install -r requirements.txt

5. En VSCode, selecciona el intérprete de Python del venv:
   Ctrl+Shift+P → "Python: Select Interpreter" → elige la ruta que contenga /venv/

## Estructura del proyecto

- app/ → Aplicación (Streamlit) y Backend REST API (FastAPI) que sirve el modelo
- data/raw/ → Dataset original sin procesar
- data/processed/ → Dataset limpio, listo para modelar
- models/ → Modelos entrenados guardados (.pkl / .joblib)
- notebooks/ → Notebooks de EDA, preprocesamiento y modelado
- src/ → Código reutilizable (funciones de preprocesamiento, etc.)
- tests/ → Tests unitarios

## Flujo de trabajo con Git

1. Nunca trabajar directo en main ni en develop.
2. Crear tu rama: git checkout -b feature/nombre-de-tu-tarea
3. Antes de empezar a trabajar cada día:
   git checkout develop
   git pull origin develop
   git checkout feature/tu-rama
   git merge develop
4. Al terminar tu parte: commit + push a tu rama, y abrir un Pull Request hacia develop (nunca hacia main).
5. Antes de cualquier commit, comprobar en qué rama estás:
   git branch

## Si añades una librería nueva

Después de instalarla con pip install, actualiza el archivo compartido:
pip freeze > requirements.txt
Y súbelo para que el resto del equipo la tenga también.
