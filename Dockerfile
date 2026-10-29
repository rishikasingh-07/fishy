FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt fetch_backbone.py ./
RUN pip install --no-cache-dir -r requirements.txt && python fetch_backbone.py
COPY . .
ENV PORT=7860
CMD gunicorn app:app --bind 0.0.0.0:${PORT} --workers 1 --threads 2 --timeout 180
