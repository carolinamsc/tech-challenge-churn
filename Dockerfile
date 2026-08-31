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
    && mkdir -p data/raw \
    && python -c "from urllib.request import urlretrieve; urlretrieve('https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv', 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')" \
    && python -m src.models.train \
    && rm -rf data/raw

EXPOSE 8501 8000

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
