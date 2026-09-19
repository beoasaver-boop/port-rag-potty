#!/usr/bin/env bash
# =============================================================================
# RAG portable · setup.sh
#
# Instala el RAG en un proyecto y lo deja indexado, todo a través de Docker:
#   1. Localiza el repositorio del RAG (este repo, o clona $RAG_REPO)
#   2. Copia rag/ + plantilla AGENTS.md al proyecto destino
#   3. Genera docker-compose.rag.yml (servicios ollama + rag)
#   4. Arranca Ollama en contenedor y pullea el modelo de embeddings
#   5. Arranca el RAG, lanza la indexación y espera a que termine (drift 0)
#
# Uso:
#   ./setup.sh [<proyecto-destino>] [--force]
#      <proyecto-destino>   ruta al proyecto a indexar       (default: ".")
#      --force              reindex force=true (re-embebe todo; LENTO en CPU)
#      --dry-run            genera compose y lo valida (docker compose config)
#                           sin arrancar contenedores ni descargar nada
#
# Variables opcionales:
#   RAG_REPO          URL git del repo RAG. Solo se necesita cuando este script
#                     NO está dentro del repositorio: se clona a un dir temporal.
#   RAG_PORT          puerto host del RAG        (default 8765)
#   RAG_EMBED_MODEL   modelo de embeddings       (default bge-m3)
#   RAG_EMBED_DIM     dimensión del vector       (default 1024)
#   RAG_COLLECTION    override de la colección Chroma
# =============================================================================
set -euo pipefail

# --- helpers -----------------------------------------------------------------
log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

jget() { # extrae un campo de JSON (bool|string|número). leer de stdin.
  local field="$1"
  grep -o "\"$field\"[[:space:]]*:[[:space:]]*[^,}]*" | head -1 \
    | sed -E 's/.*:[[:space:]]*//; s/^"(.*)"$/\1/'
}

url_slug() { # basename → slug seguro para nombres de contenedor/volumen
  local b
  b="$(basename "$1")"
  b="${b// /-}"
  b="$(printf '%s' "$b" | tr -c 'a-zA-Z0-9_-' '_' | tr 'A-Z' 'a-z')"
  b="${b#_}"; b="${b%_}"
  [ -n "$b" ] && printf '%s' "$b" || printf 'proyecto'
}

wait_http() { # url [intentos] [intervalo] — espera HTTP 200
  local url="$1" n="${2:-90}" iv="${3:-2}" i
  for i in $(seq 1 "$n"); do
    if curl -fsS --max-time 3 "$url" >/dev/null 2>&1; then return 0; fi
    sleep "$iv"
  done
  return 1
}

wait_healthy() { # nombre-contenedor [intentos] [intervalo] — espera healthcheck
  local c="$1" n="${2:-60}" iv="${3:-2}" st i
  for i in $(seq 1 "$n"); do
    st="$(docker inspect --format '{{.State.Health.Status}}' "$c" 2>/dev/null || printf starting)"
    [ "$st" = healthy ] && return 0
    [ "$st" = unhealthy ] && die "el contenedor $c quedó unhealthy"
    sleep "$iv"
  done
  return 1
}

usage() {
  sed -n 's/^# \{0,1\}//p' "$0" | sed -n '3,18p'
}

# --- argumentos --------------------------------------------------------------
DEST="." ; FORCE=0 ; DRY=0
for a in "$@"; do
  case "$a" in
    --force)   FORCE=1 ;;
    --dry-run) DRY=1   ;;
    -h|--help) usage; exit 0 ;;
    *)         DEST="$a" ;;
  esac
done

# --- prerequisitos -----------------------------------------------------------
command -v docker >/dev/null 2>&1 || die "docker no está instalado"
docker compose version >/dev/null 2>&1 \
  || docker-compose --version >/dev/null 2>&1 \
  || die "docker compose (v2) no está disponible"
command -v curl >/dev/null 2>&1 || die "curl no está instalado"

PORT="${RAG_PORT:-8765}"
MODEL="${RAG_EMBED_MODEL:-bge-m3}"
EMBED_DIM="${RAG_EMBED_DIM:-1024}"
COMPOSE_NAME="docker-compose.rag.yml"

# --- localizar el repo del RAG ------------------------------------------------
# Si este script está dentro del repositorio (rag/server.py al lado) se usa tal
# cual; si no, se clona $RAG_REPO a un dir temporal.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/rag/server.py" ] && [ -f "$SCRIPT_DIR/rag/config.py" ]; then
  RAG_SRC="$SCRIPT_DIR"
  TMP_RAG=""
  log "repositorio del RAG: $RAG_SRC"
elif [ -n "${RAG_REPO:-}" ]; then
  TMP_RAG="$(mktemp -d)"
  trap 'rm -rf "$TMP_RAG"' EXIT
  log "clonando $RAG_REPO ..."
  git clone --quiet --depth 1 "$RAG_REPO" "$TMP_RAG/rag-portable"
  RAG_SRC="$TMP_RAG/rag-portable"
else
  die "no encuentro el repo del RAG. Copia setup.sh dentro de RAG_portable, o exporta RAG_REPO=<url-git>"
fi

# --- proyecto destino ---------------------------------------------------------
[ -d "$DEST" ] || mkdir -p "$DEST" || die "no se pudo crear $DEST"
DEST="$(cd "$DEST" && pwd)"
SLUG="$(url_slug "$DEST")"
PROJECT="$(basename "$DEST")"

log "proyecto: $DEST (slug=$SLUG, puerto=$PORT, modelo=$MODEL)"

# --- copiar motor -------------------------------------------------------------
if [ -f "$DEST/rag/requirements.txt" ]; then
  log "rag/ ya existe en $DEST — actualizando código"
fi
mkdir -p "$DEST/rag/data"
cp -R "$RAG_SRC/rag/." "$DEST/rag/"
[ -f "$DEST/AGENTS.md" ] || { [ -f "$RAG_SRC/AGENTS.md" ] && cp "$RAG_SRC/AGENTS.md" "$DEST/AGENTS.md" && log "plantilla AGENTS.md copiada"; }

# --- generar docker-compose.rag.yml ------------------------------------------
cat > "$DEST/$COMPOSE_NAME" <<YAML
name: rag-${SLUG}
services:
  ollama:
    image: ollama/ollama:latest
    container_name: rag-${SLUG}-ollama
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "/bin/sh", "-c", "ollama list >/dev/null 2>&1 || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5
    volumes:
      - ollama_data:/root/.ollama

  rag:
    build: ./rag
    image: rag-${SLUG}-rag
    container_name: rag-${SLUG}-rag
    restart: unless-stopped
    ports:
      - "${PORT}:8765"
    environment:
      OLLAMA_HOST: http://ollama:11434
      RAG_PORT: 8765
      RAG_EMBED_MODEL: ${MODEL}
      RAG_EMBED_DIM: ${EMBED_DIM}
      RAG_PROJECT: ${PROJECT}
      RAG_COLLECTION: ${RAG_COLLECTION:-${SLUG}_chunks}
    volumes:
      - .:/app
      - rag_data:/app/rag/data
    working_dir: /app
    depends_on:
      ollama:
        condition: service_healthy

volumes:
  ollama_data:
  rag_data:
YAML
log "generado $COMPOSE_NAME"

# --- modo dry-run: validar sin arrancar nada ----------------------------------
if [ "$DRY" -eq 1 ]; then
  log "dry-run — validando compose (sin descargas ni contenedores)"
  docker compose -f "$DEST/$COMPOSE_NAME" config >/dev/null \
    || die "compose inválido"
  log "compose válido: $DEST/$COMPOSE_NAME"
  exit 0
fi

# --- arrancar Ollama y pullear el modelo --------------------------------------
log "arrancando Ollama (healthcheck)..."
docker compose -f "$DEST/$COMPOSE_NAME" up -d ollama
wait_healthy "rag-${SLUG}-ollama" || die "Ollama no arrancó a tiempo"
log "pulleando modelo $MODEL ..."
docker exec "rag-${SLUG}-ollama" ollama pull "$MODEL"

# --- arrancar el RAG ----------------------------------------------------------
log "arrancando el servicio RAG ..."
docker compose -f "$DEST/$COMPOSE_NAME" up -d --build rag
wait_http "http://localhost:${PORT}/health" || die "el RAG no responde en :${PORT}"

# --- indexar y esperar --------------------------------------------------------
log "lanzando reindex (incremental)..."
if [ "$FORCE" -eq 1 ]; then
  log "force=true — re-embebiendo todo (puede tardar mucho en CPU)"
  curl -fsS -X POST "http://localhost:${PORT}/reindex?force=true" >/dev/null
else
  curl -fsS -X POST "http://localhost:${PORT}/reindex" >/dev/null
fi

log "esperando a que termine la indexación ..."
sleep 2
st="" ; running=""
while true; do
  st="$(curl -fsS "http://localhost:${PORT}/stats")"
  running="$(printf '%s' "$st" | jget running)"
  [ "$running" = false ] && break
  rstat="$(printf '%s' "$st" | jget status)"
  [ "$rstat" = error ] && die "la indexación falló (mira /stats)"
  sleep 3
done

CHROMA="$(printf '%s' "$st" | jget chroma_chunks)"
MANIFEST="$(printf '%s' "$st" | jget manifest_chunks)"
DRIFT="$(printf '%s' "$st" | jget drift)"
STATUS="$(printf '%s' "$st" | jget status)"

log "índice: manifest=$MANIFEST chroma=$CHROMA drift=$DRIFT status=$STATUS"
if [ "${DRIFT:-0}" != "0" ]; then
  die "drift != 0 — repite: curl -X POST http://localhost:${PORT}/reindex"
fi

# --- resumen ------------------------------------------------------------------
cat <<SUMMARY
========================================================================
  RAG listo en $DEST
  Lanza el servicio:   docker compose -f $DEST/$COMPOSE_NAME up -d
  Health:              curl http://localhost:${PORT}/health
  Indexa cambios:      curl -X POST http://localhost:${PORT}/reindex
  Stats:               curl http://localhost:${PORT}/stats
  (Re)config manual:   editar $DEST/rag.config.json y reiniciar
========================================================================
SUMMARY