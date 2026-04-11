FROM python:3.11-slim

ENV MALLOC_ARENA_MAX=2

WORKDIR /app

# CACHEBUST=3: force fresh pip install
ARG CACHEBUST=3
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/

# Use Python to read PORT — same pattern as JARVIS (os.environ.get)
CMD ["python", "-c", "import os, uvicorn; uvicorn.run('api.main:app', host='0.0.0.0', port=int(os.environ.get('PORT', 8000)), log_level='info')"]
