FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader stopwords punkt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "workflow_pipeline:app", "--host", "0.0.0.0", "--port", "8000"]
