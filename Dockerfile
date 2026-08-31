FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY src ./src
COPY app.py ./app.py
COPY data ./data
COPY models ./models

RUN pip install --no-cache-dir . \
    && python -m src.data.download \
    && python -m src.models.train \
    && rm -rf data/raw

EXPOSE 8501 8000

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
