FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/
COPY src/ ./src/
COPY .streamlit/ ./.streamlit/

EXPOSE 8501
EXPOSE 8000

ENV PYTHONPATH=/app

HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

CMD ["sh", "-c", "streamlit run app/main.py --server.port=8501 --server.address=0.0.0.0 & uvicorn app.api:app --host 0.0.0.0 --port 8000 && wait"]
