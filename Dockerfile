FROM python:3.11-slim

WORKDIR /app

# OpenCV va tizim kutubxonalari
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000
ENV HOST=0.0.0.0

EXPOSE 8000

CMD ["python", "server.py"]
