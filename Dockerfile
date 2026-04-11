FROM python:3.11-slim
ENV MALLOC_ARENA_MAX=2
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn
COPY test_main.py .
CMD sh -c "uvicorn test_main:app --host 0.0.0.0 --port ${PORT:-8000}"
