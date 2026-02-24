FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir -r requirements.txt

ENV NLTK_DATA=/usr/local/nltk_data
RUN mkdir -p /usr/local/nltk_data
RUN python -m nltk.downloader -d /usr/local/nltk_data stopwords punkt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "workflow_pipeline:app", "--host", "0.0.0.0", "--port", "8000"]
