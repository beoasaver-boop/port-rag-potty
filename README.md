# port-rag-potty

# RAG portable

> **RAG con ChromaDB + Ollama, «cópialo y funciona».** Sirve a tus agentes de
> IA (opencode, Claude, Cursor, Cline…) contexto curado de cualquier proyecto:
> reduce de ~80-150k tokens de lectura de archivos a ~3-5k de respuestas HTTP
> precisas.

Un directorio `rag/` + una plantilla `AGENTS.md` que se copian dentro de
**cualquier** proyecto. No está atado a ningún tipo de código (frontend, Python,
docs…) — la configuración por proyecto vive en un único `rag.config.json`
auto-generado.

---

## Qué resuelve

Un agente nuevo que entra a un repo desconocido hoy se lee README + archivos al
completo (80-150k tokens). Con el RAG hace 4-5 llamadas HTTP (~3-5k tokens) y
tiene:

- **`/methodology`** — el «alma» del proyecto: propósito, arquitectura,
  convenciones (chunks de README/AGENTS marcados como methodology).
- **`/query`** — búsqueda semántica sobre todo el código (embeddings bge-m3).
- **`/categories`** — grupos (temas CSS, módulos…) y sus tokens/archivos (opcional).

---

## Requisitos

- **Docker** (para la instalación automatizada) **o** Python 3.10+ (instalación manual)
- Ollama con el modelo de embeddings, por ejemplo `bge-m3`:
  ```bash
  ollama pull bge-m3
  ```
  La instalación automatizada usa un contenedor `ollama/ollama` y pullea el
  modelo por ti.

---

## Instalación

El repo trae scripts de automatización equivalente en **bash** (`setup.sh`) y
**PowerShell** (`setup.ps1`): clonan/buscan el repositorio del RAG, lo copian
al proyecto destino, montan los contenedores (Ollama + RAG), pullean el modelo,
crean la base Chroma e indexan — todo hasta que `/stats` da `drift = 0`.

### Opción A — Automatizada (recomendada, Docker)

```bash
# 1) Clona el repositorio del RAG (solo la primera vez)
git clone <url-de-este-repo> rag-portable
cd rag-portable

# 2) Instala el RAG en tu proyecto y déjalo indexado
./setup.sh /ruta/al/proyecto        # bash (Linux/macOS/WSL)
# o en Windows PowerShell:
.\setup.ps1 C:\ruta\al\proyecto
```

Parámetros y variables (en ambos scripts):

| Flag / variable | Descripción | Default |
|---|---|---|
| `<proyecto>` | Directorio a indexar | `.` |
| `--force` | Reindex `force=true` (re-embebe todo; **lento** en CPU) | off |
| `RAG_PORT` | Puerto host del RAG | `8765` |
| `RAG_EMBED_MODEL` | Modelo de embeddings a pullear/usar | `bge-m3` |
| `RAG_EMBED_DIM` | Dimensión del vector | `1024` |
| `RAG_COLLECTION` | Override del nombre de colección Chroma | `<proyecto>_chunks` |

Qué hace exactamente:

1. Copia `rag/` (+ plantilla `AGENTS.md`, si el proyecto no tiene una) al destino.
2. Genera `docker-compose.rag.yml` (servicios `ollama` + `rag`).
3. Arranca Ollama en contenedor y pullea `bge-m3`.
4. Arranca el RAG en `http://localhost:8765`.
5. Lanza `POST /reindex` y espera a que termine (`drift == 0`).

Después de tocar archivos, re-indexa con:

```bash
curl -X POST 'http://localhost:8765/reindex'
```

### Opción B — Manual (sin Docker)

```bash
cd /ruta/al/proyecto
cp -r <este-repo>/rag ./rag                     # copia el motor
# opcional: plantilla de instrucciones para agentes
cp <este-repo>/AGENTS.md ./AGENTS.md            # solo si no existe la tuya

pip install -r rag/requirements.txt
python -m rag.server &                          # sirve en :8765
curl -X POST 'http://localhost:8765/reindex'    # indexa en background
curl -s http://localhost:8765/stats             # drift debe ser 0
```

La primera indexación tarda 1-5 min (depende del repo y de si Ollama tiene el
modelo en memoria). Un `force=true` re-embebe todo y en CPU puede tardar >1h.

---

## Configuración (opcional)

En el primer arranque se genera `rag.config.json` en la raíz del proyecto con
valores detectados automáticamente (nombre de colección, globs, exclusiones,
methodology y si hay que activar categorías). Se recarga reiniciando el servidor.

```jsonc
{
  "project_name": "mi-proyecto",
  "collection": "mi_proyecto_chunks",
  "index_globs": ["**/*.md", "**/*.css", "**/*.js", "**/*.py", "..."],
  "exclude_dirs": ["node_modules", ".git", "venv", "...", "data", "rag/data"],
  "methodology": { "entire_files": ["AGENTS.md"], "heading_patterns": [] },
  "graph": { "block_regex": "[data-theme='...'] {...}", "token_regex": "--x:", "..." }
}
```

Documentación completa de cada campo: `rag/README.md`.

Variables de entorno: `OLLAMA_HOST`, `RAG_EMBED_MODEL`, `RAG_EMBED_DIM`,
`RAG_HOST`, `RAG_PORT`, `RAG_LOG_LEVEL`, `RAG_PROJECT`, `RAG_COLLECTION`.

---

## Endpoints

| Método | Endpoint | Uso |
|---|---|---|
| `GET` | `/health` | Estado Ollama + colección + grafo |
| `GET` | `/methodology` | «Alma» del proyecto (inyectar SIEMPRE al agente) |
| `GET` | `/categories` / `/category/{id}` | Categorías y su ficha (tokens+archivos) |
| `POST` | `/query` | Búsqueda semántica `{q, k, category?, file_glob?}` |
| `GET` | `/related/{kind}/{id}` | Vecinos en el grafo (`category`/`file`/`token`) |
| `POST` | `/reindex` | Re-indexar (incremental + reconciliación) |
| `GET` | `/stats` | Estadísticas + `drift` + estado del reindex |

---

## Estructura del repo

```
AGENTS.md             ← instrucciones obligatorias para agentes de IA (plantilla)
README.md             ← este archivo
setup.sh / setup.ps1  ← instalación automatizada vía Docker
rag/
  server.py           ← API HTTP (FastAPI)
  indexer.py          ← chunk + embed + upsert + reconciliación
  chunker.py          ← parseo semántico por tipo de archivo
  graph.py            ← grafo categorías↔tokens↔archivos (JSON)
  embeddings.py       ← cliente Ollama (bge-m3)
  config.py           ← configuración + bootstrap de rag.config.json
  requirements.txt    ← dependencias Python
  Dockerfile          ← imagen del servicio
  README.md           ← documentación técnica del motor
```

---

## Licencia privada de uso

Uso personal. Sujétalo a los términos de tu organización si lo compartes.