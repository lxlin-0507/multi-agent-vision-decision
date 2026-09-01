FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 7861

CMD ["sh", "-c", "APP_PORT=${PORT:-7861} GRADIO_SERVER_NAME=0.0.0.0 python app.py"]
