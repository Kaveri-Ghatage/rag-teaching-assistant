FROM python:3.10-slim

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project (embeddings.joblib and faiss_index.bin are included)
COPY . .

# Ollama runs as a separate container (see docker-compose.yml)
ENV OLLAMA_URL=http://ollama:11434

CMD ["python", "inference.py"]
