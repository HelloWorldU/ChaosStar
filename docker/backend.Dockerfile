FROM python:3.11.12-slim AS base

WORKDIR /app

COPY web/backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install playwright && \
    playwright install --with-deps chromium

COPY app/ ./app/
COPY config/ ./config/
COPY web/backend/ ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
