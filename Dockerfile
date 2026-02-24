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

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Setup NLTK data
ENV NLTK_DATA=/usr/local/nltk_data
RUN mkdir -p /usr/local/nltk_data
RUN python -m nltk.downloader -d /usr/local/nltk_data stopwords punkt punkt_tab

COPY . .

EXPOSE 8000

CMD ["uvicorn", "workflow_pipeline:app", "--host", "0.0.0.0", "--port", "8000"]
