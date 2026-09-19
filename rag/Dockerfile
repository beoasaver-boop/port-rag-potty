# RAG portable · imagen Docker (genérico para cualquier proyecto)
#
# Contexto de build = directorio rag/ (build: ./rag en el compose):
#   docker build -t rag .
#
# En runtime se monta el código real del proyecto en /app (volumen .:/app),
# con el motor en /app/rag y su Chroma en /app/rag/data (volumen rag_data).
# /app/rag.config.json se genera solo en el primer arranque.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/rag/
RUN pip install --no-cache-dir -r /app/rag/requirements.txt

COPY . /app/rag/

EXPOSE 8765

# Healthcheck para orquestadores
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fs http://localhost:8765/health >/dev/null || exit 1

CMD ["python", "-m", "rag.server"]