FROM python:3.11-slim

# MALLOC_ARENA_MAX=2 — ОБЯЗАТЕЛЬНО для Python с C extensions (asyncpg, httpx)
# Ограничивает glibc арены памяти, иначе Railway OOM на малых инстансах
ENV MALLOC_ARENA_MAX=2

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/

EXPOSE 8000

CMD sh -c "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"
