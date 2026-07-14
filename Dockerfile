# Use the slim Debian-based image for smaller footprint and ChromaDB SQLite compatibility
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Layer 1: Cache and install dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Layer 2: Copy the application code, PDFs, and evaluation schemas
COPY . .

# Expose the API and WebSocket port
EXPOSE 8001

# Run strictly with 1 worker to preserve InMemorySessionService state
CMD ["uvicorn", "mock_server:app", "--host", "0.0.0.0", "--port", "8001"]
