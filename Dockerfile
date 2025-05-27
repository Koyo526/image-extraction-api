FROM python:3.10-slim

COPY .env /app/.env
WORKDIR /app

COPY ./app /app

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


