<#
  RAG portable · setup.ps1
  ----------------------------------------------------------------------------
  Instala el RAG en un proyecto y lo deja indexado, todo a través de Docker:

    1. Localiza el repositorio del RAG (este repo, o clona $env:RAG_REPO)
    2. Copia rag/ + plantilla AGENTS.md al proyecto destino
    3. Genera docker-compose.rag.yml (servicios ollama + rag)
    4. Arranca Ollama en contenedor y pullea el modelo de embeddings
    5. Arranca el RAG, lanza la indexación y espera a que termine (drift 0)

  Uso:
    .\setup.ps1 [<proyecto-destino>] [-Force]
       <proyecto-destino>   ruta al proyecto a indexar    (default: ".")
       -Force               reindex force=true (re-embebe todo; LENTO en CPU)
       -DryRun              genera compose y lo valida (docker compose config)
                            sin arrancar contenedores ni descargar nada

  Variables opcionales (env):
    RAG_REPO          URL git del repo RAG. Solo se necesita cuando este script
                      NO está dentro del repositorio: se clona a un dir temporal.
    RAG_PORT          puerto host del RAG       (default 8765)
    RAG_EMBED_MODEL   modelo de embeddings      (default bge-m3)
    RAG_EMBED_DIM     dimensión del vector      (default 1024)
    RAG_COLLECTION    override de la colección Chroma
  (RAG_PROJECT se deriva del nombre del proyecto destino)
#>

param(
  [string]$Dest = ".",
  [switch]$Force,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function ConvertTo-Slug([string]$s) {
  $b = [Regex]::Replace((Split-Path -Leaf $s), '[^a-zA-Z0-9_-]', '_')
  if ([string]::IsNullOrWhiteSpace($b)) { return 'proyecto' }
  return $b.ToLower()
}

# --- prerequisitos ------------------------------------------------------------
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
  throw "docker no esta instalado"
}
docker compose version *> $null
if ($LASTEXITCODE -ne 0) { throw "docker compose (v2) no esta disponible" }

$Port    = if ($env:RAG_PORT)    { $env:RAG_PORT }    else { 8765 }
$Model   = if ($env:RAG_EMBED_MODEL) { $env:RAG_EMBED_MODEL } else { "bge-m3" }
$EmbedDim= if ($env:RAG_EMBED_DIM)   { $env:RAG_EMBED_DIM }   else { 1024 }
$ComposeName = "docker-compose.rag.yml"

# --- localizar el repo del RAG -------------------------------------------------
# Si este script esta dentro del repositorio (rag/server.py al lado) se usa tal
# cual; si no, se clona $env:RAG_REPO a un dir temporal.
$RagSrc = $PSScriptRoot
$TmpRag = $null
if ((Test-Path "$RagSrc/rag/server.py") -and (Test-Path "$RagSrc/rag/config.py")) {
  Write-Step "repositorio del RAG: $RagSrc"
} elseif ($env:RAG_REPO) {
  $TmpRag = Join-Path $env:TEMP ("rag_portable_" + [guid]::NewGuid().ToString('n'))
  Write-Step "clonando $($env:RAG_REPO) ..."
  git clone --quiet --depth 1 $env:RAG_REPO $TmpRag
  if ($LASTEXITCODE -ne 0) { throw "no se pudo clonar $($env:RAG_REPO)" }
  $RagSrc = $TmpRag
} else {
  throw "no encuentro el repo del RAG. Copia setup.ps1 dentro de RAG_portable, o setea RAG_REPO=<url-git>"
}

try {
  # --- proyecto destino ---------------------------------------------------------
  if (-not (Test-Path $Dest)) { New-Item -ItemType Directory -Path $Dest -Force | Out-Null }
  $Dest  = (Resolve-Path $Dest).Path
  $Slug  = ConvertTo-Slug $Dest

  Write-Step "proyecto: $Dest (slug=$Slug, puerto=$Port, modelo=$Model)"

  # --- copiar motor -------------------------------------------------------------
  if (Test-Path "$Dest/rag/requirements.txt") { Write-Step "rag/ ya existe en $Dest - actualizando codigo" }
  New-Item -ItemType Directory -Path "$Dest/rag/data" -Force | Out-Null
  Copy-Item -Path "$RagSrc/rag/*" -Destination "$Dest/rag" -Recurse -Force
  if (-not (Test-Path "$Dest/AGENTS.md")) {
    if (Test-Path "$RagSrc/AGENTS.md") {
      Copy-Item "$RagSrc/AGENTS.md" "$Dest/AGENTS.md"
      Write-Step "plantilla AGENTS.md copiada"
    }
  }

  # --- generar docker-compose.rag.yml --------------------------------------------
  $envLines = [System.Collections.Generic.List[string]]::new()
  $envLines.Add("      OLLAMA_HOST: http://ollama:11434") | Out-Null
  $envLines.Add("      RAG_PORT: 8765") | Out-Null
  $envLines.Add("      RAG_EMBED_MODEL: $Model") | Out-Null
  $envLines.Add("      RAG_EMBED_DIM: $EmbedDim") | Out-Null
  $envLines.Add("      RAG_PROJECT: $((Split-Path -Leaf $Dest))") | Out-Null
  if ($env:RAG_COLLECTION) {
    $envLines.Add("      RAG_COLLECTION: $($env:RAG_COLLECTION)") | Out-Null
  } else {
    $envLines.Add("      RAG_COLLECTION: ${Slug}_chunks") | Out-Null
  }
  $envBlock = ($envLines -join "`n")

  $compose = @"
name: rag-${Slug}
services:
  ollama:
    image: ollama/ollama:latest
    container_name: rag-${Slug}-ollama
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
    image: rag-${Slug}-rag
    container_name: rag-${Slug}-rag
    restart: unless-stopped
    ports:
      - "${Port}:8765"
    environment:
$envBlock
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
"@
  Set-Content -Path "$Dest/$ComposeName" -Value $compose -Encoding utf8
  Write-Step "generado $ComposeName"

  if ($DryRun) {
    Write-Step "dry-run - validando compose (sin descargas ni contenedores)"
    docker compose -f "$Dest/$ComposeName" config *> $null
    if ($LASTEXITCODE -ne 0) { throw "compose invalido" }
    Write-Step "compose valido: $Dest/$ComposeName"
    return
  }

  # --- arrancar Ollama y pullear el modelo ----------------------------------------
  Write-Step "arrancando Ollama (healthcheck)..."
  docker compose -f "$Dest/$ComposeName" up -d ollama
  if ($LASTEXITCODE -ne 0) { throw "no se pudo arrancar ollama" }

  $oc = "rag-${Slug}-ollama"
  for ($i = 0; $i -lt 60; $i++) {
    $st = docker inspect --format '{{.State.Health.Status}}' $oc 2>$null
    if (-not $st) { $st = "starting" }
    if ($st -eq "healthy") { break }
    if ($st -eq "unhealthy") { throw "el contenedor $oc quedo unhealthy" }
    Start-Sleep -Seconds 2
  }
  if ($i -ge 60) { throw "Ollama no arranco a tiempo" }

  Write-Step "pulleando modelo $Model ..."
  docker exec $oc ollama pull $Model

  # --- arrancar el RAG -------------------------------------------------------------
  Write-Step "arrancando el servicio RAG ..."
  docker compose -f "$Dest/$ComposeName" up -d --build rag
  if ($LASTEXITCODE -ne 0) { throw "no se pudo arrancar el RAG" }

  $health = "http://localhost:${Port}/health"
  $ok = $false
  for ($i = 0; $i -lt 90; $i++) {
    try {
      $r = Invoke-WebRequest -Uri $health -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
      if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch { }
    Start-Sleep -Seconds 2
  }
  if (-not $ok) { throw "el RAG no responde en $health" }

  # --- indexar y esperar -----------------------------------------------------------
  Write-Step "lanzando reindex (incremental)..."
  $uri = "http://localhost:${Port}/reindex"
  if ($Force) {
    Write-Step "force=true - re-embebiendo todo (puede tardar mucho en CPU)"
    $uri += "?force=true"
  }
  Invoke-RestMethod -Method Post -Uri $uri -ErrorAction Stop | Out-Null

  Write-Step "esperando a que termine la indexacion ..."
  Start-Sleep -Seconds 2
  $st = $null
  do {
    Start-Sleep -Seconds 3
    $st = Invoke-RestMethod -Uri "http://localhost:${Port}/stats" -ErrorAction Stop
    if ($st.reindex.status -eq "error") { throw "la indexacion fallo (mira /stats)" }
  } while ($st.reindex.running)

  $drift = $st.drift
  Write-Step "indice: manifest=$($st.manifest_chunks) chroma=$($st.chroma_chunks) drift=$drift status=$($st.reindex.status)"
  if ($drift -ne 0) { throw "drift != 0 - repite: curl -X POST http://localhost:${Port}/reindex" }

  # --- resumen --------------------------------------------------------------------
  Write-Host @"
========================================================================
  RAG listo en $Dest
  Lanza el servicio:    docker compose -f $Dest/$ComposeName up -d
  Health:               Invoke-RestMethod http://localhost:${Port}/health
  Indexa cambios:       Invoke-RestMethod -Method Post http://localhost:${Port}/reindex
  Stats:                Invoke-RestMethod http://localhost:${Port}/stats
  (Re)config manual:    editar $Dest/rag.config.json y reiniciar
========================================================================
"@ -ForegroundColor Green
}
finally {
  if ($TmpRag -and (Test-Path $TmpRag)) { Remove-Item -Recurse -Force $TmpRag }
}