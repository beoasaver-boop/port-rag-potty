"""RAG portable · grafo ligero del proyecto (genérico).

Estructura (graph.json):
  {
    "categories": {
      "<id>": { "files": [...], "tokens": [...], "scripts": [...] }
    },
    "tokens": { "--bg": { "defined_by": ["cauldron", ...] } },
    "files": { "css/themes/cauldron.css": { "categories": ["cauldron"], "imports": [...] } },
    "edges": [ {"from": "category:cauldron", "rel": "defines_token", "to": "token:--bg"}, ... ]
  }

Es 100% configurable vía `rag.config.json → graph`:
  · block_regex      → categorías definidas en bloques (grupo 1=cat, 2=cuerpo).
  · token_regex      → tokens dentro del bloque → edge category → token.
  · mention_regex    → menciones sueltas de categoría (data-theme, referencias...).
  · js_category_regex→ heurística JS → categorías (controles de UI, etc.).
  · summary_file     → archivo con resúmenes 1-línea por categoría (regex con {id}).

Si todas las regex van vacías (proyecto no-CSS), el grafo queda solo con
categorías vacías y los endpoints /categories devuelven lista vacía: el RAG
sigue siendo útil por vectores + methodology + file-imports.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from . import chunker
from . import config


# ── Modelo simple (JSON-serializable) ────────────────────────────────────────
class Graph:
    """Grafo ligero, persistido como dict JSON."""

    def __init__(self) -> None:
        self.categories: dict[str, dict] = {}
        self.tokens: dict[str, dict] = {}
        self.files: dict[str, dict] = {}
        self.edges: list[dict] = []
        self.category_summaries: dict[str, str] = {}

    # ── mutadores ──
    def ensure_category(self, cid: str) -> dict:
        sanitized = cid.strip()[:80]
        if not sanitized:
            raise ValueError("categoría vacía")
        self.categories.setdefault(sanitized, {"files": set(), "tokens": set(), "scripts": set()})
        return self.categories[sanitized]

    def ensure_token(self, tok: str) -> dict:
        self.tokens.setdefault(tok, {"defined_by": set()})
        return self.tokens[tok]

    def ensure_file(self, rel: str) -> dict:
        self.files.setdefault(rel, {"categories": set(), "imports": set()})
        return self.files[rel]

    def add_edge(self, src: str, rel: str, dst: str) -> None:
        if any(e["from"] == src and e["rel"] == rel and e["to"] == dst for e in self.edges):
            return
        self.edges.append({"from": src, "rel": rel, "to": dst})

    # ── serialización ──
    def to_dict(self) -> dict:
        return {
            "categories": {
                k: {
                    "files": sorted(v["files"]),
                    "tokens": sorted(v["tokens"]),
                    "scripts": sorted(v["scripts"]),
                }
                for k, v in self.categories.items()
            },
            "tokens": {
                k: {"defined_by": sorted(v["defined_by"])}
                for k, v in self.tokens.items()
            },
            "files": {
                k: {
                    "categories": sorted(v["categories"]),
                    "imports": sorted(v["imports"]),
                }
                for k, v in self.files.items()
            },
            "edges": [
                {"from": e["from"], "rel": e["rel"], "to": e["to"]}
                for e in self.edges
            ],
            "category_summaries": self.category_summaries,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Graph":
        g = cls()
        for k, v in d.get("categories", {}).items():
            g.categories[k] = {
                "files": set(v.get("files", [])),
                "tokens": set(v.get("tokens", [])),
                "scripts": set(v.get("scripts", [])),
            }
        for k, v in d.get("tokens", {}).items():
            g.tokens[k] = {"defined_by": set(v.get("defined_by", []))}
        for k, v in d.get("files", {}).items():
            g.files[k] = {
                "categories": set(v.get("categories", [])),
                "imports": set(v.get("imports", [])),
            }
        g.edges = list(d.get("edges", []))
        g.category_summaries = dict(d.get("category_summaries", {}))
        return g

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Graph":
        if not path.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            return cls()


# ── Construir grafo desde el proyecto ──────────────────────────────────────
_HTML_SCRIPT = re.compile(r'<script\s+src=["\']([^"\']+)["\']', re.IGNORECASE)
_HTML_LINK = re.compile(r'<link\s+[^>]*href=["\']([^"\']+\.css)["\']', re.IGNORECASE)

_JS_EXTS = frozenset({"js", "mjs", "cjs", "ts", "tsx", "jsx"})


def _summary_for_category(category_id: str, project_root: Path) -> str:
    """Resumen de 1 línea para una categoría desde un archivo declarativo."""
    summary_file = config.GRAPH_SUMMARY_FILE
    summary_regex = config.GRAPH_SUMMARY_REGEX
    if not summary_file or not summary_regex:
        return ""
    path = project_root / summary_file
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return ""
    pattern = re.compile(summary_regex.replace("{id}", re.escape(category_id)), re.DOTALL)
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


def build_graph(project_root: Path, chunks: Iterable | None = None) -> Graph:
    g = Graph()

    # 1) Recorrer archivos indexados: bloques, menciones, tokens
    for p in chunker.iter_project_files(project_root):
        ext = p.suffix.lower().lstrip(".")
        rel = str(p.relative_to(project_root))
        try:
            content = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        if ext in config.GRAPH_BLOCK_EXTS and config.GRAPH_BLOCK_REGEX:
            for m in re.finditer(config.GRAPH_BLOCK_REGEX, content):
                cid = (m.group(1) or "").strip()
                if not cid:
                    continue
                body = m.group(2) or ""
                cat = g.ensure_category(cid)
                cat["files"].add(rel)
                g.ensure_file(rel)["categories"].add(cid)
                g.add_edge(f"category:{cid}", "defined_in", f"file:{rel}")
                if config.TOKEN_REGEX:
                    for tok in set(re.findall(config.TOKEN_REGEX, body)):
                        cat["tokens"].add("--" + tok)
                        g.ensure_token("--" + tok)["defined_by"].add(cid)
                        g.add_edge(f"category:{cid}", "defines_token", f"token:--{tok}")

        if ext in config.GRAPH_MENTION_EXTS and config.CATEGORY_REGEX:
            for m in re.finditer(config.CATEGORY_REGEX, content):
                cid = (m.group(1) if m.groups() else m.group(0) or "").strip()
                if not cid:
                    continue
                cat = g.ensure_category(cid)
                cat["files"].add(rel)
                g.ensure_file(rel)["categories"].add(cid)

        if ext in config.GRAPH_TOKEN_EXTS and config.TOKEN_REGEX:
            if ext not in config.GRAPH_BLOCK_EXTS:
                # tokens fuera de bloques: edge token → archivo
                for tok in set(re.findall(config.TOKEN_REGEX, content)):
                    g.ensure_token("--" + tok)
                    g.add_edge(f"token:--{tok}", "appears_in", f"file:{rel}")

        if ext in _JS_EXTS and config.JS_CATEGORY_CHECK_REGEX:
            for m in re.finditer(config.JS_CATEGORY_CHECK_REGEX, content):
                candidate = next((gr for gr in m.groups() if gr), None)
                if not candidate:
                    continue
                tid = candidate.lower() if candidate[0].isupper() else candidate
                cat = g.ensure_category(tid)
                cat["scripts"].add(rel)
                g.add_edge(f"category:{tid}", "interacts_with", f"file:{rel}")

    # 2) Página HTML de entrada: imports de scripts/links
    entry = config.GRAPH_HTML_ENTRY
    if entry:
        base = project_root / entry
        rel_entry = str(base.relative_to(project_root)) if base.exists() else entry
        if base.exists():
            try:
                content = base.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                content = ""
            for m in _HTML_SCRIPT.finditer(content):
                src = m.group(1).lstrip("./")
                g.ensure_file(src)["imports"].add(rel_entry)
                g.add_edge(f"file:{src}", "imported_by", f"file:{rel_entry}")
            for m in _HTML_LINK.finditer(content):
                href = m.group(1).lstrip("./")
                g.ensure_file(href)["imports"].add(rel_entry)
                g.add_edge(f"file:{href}", "imported_by", f"file:{rel_entry}")

    # 3) Resúmenes
    for cid in list(g.categories.keys()):
        summary = _summary_for_category(cid, project_root)
        if summary:
            g.category_summaries[cid] = summary

    # 4) Limpiar "categorías fantasma": sin tokens ni archivos (los scripts
    #    solos no bastan para considerarla real).
    phantoms = [
        cid for cid, data in g.categories.items()
        if not data.get("tokens") and not data.get("files")
    ]
    for cid in phantoms:
        g.edges = [
            e for e in g.edges
            if not (e["from"] == f"category:{cid}" or e["to"] == f"category:{cid}")
        ]
        g.categories.pop(cid, None)

    return g


# ── Helpers de consulta ────────────────────────────────────────────────────
def neighbor_files(graph: Graph, category_id: str) -> list[str]:
    c = graph.categories.get(category_id, {})
    return sorted(set(c.get("files", set())) | set(c.get("scripts", set())))


def neighbor_tokens(graph: Graph, category_id: str) -> list[str]:
    c = graph.categories.get(category_id, {})
    return sorted(c.get("tokens", set()))


def related_entity(graph: Graph, entity: str, max_hops: int = 2) -> dict:
    """Vecinos de un nodo en 1 o 2 saltos.

    entity: 'category:cauldron' | 'file:js/cauldron.js' | 'token:--bg'
    """
    out: dict[str, set] = {"neighbors": set()}
    if entity.startswith("category:"):
        cid = entity[len("category:"):]
        if cid in graph.categories:
            c = graph.categories[cid]
            for f in c.get("files", set()):
                out["neighbors"].add(f"file:{f}")
            for s in c.get("scripts", set()):
                out["neighbors"].add(f"file:{s}")
            for tk in c.get("tokens", set()):
                out["neighbors"].add(f"token:{tk}")
    elif entity.startswith("file:"):
        rel = entity[len("file:"):]
        if rel in graph.files:
            f = graph.files[rel]
            for c in f.get("categories", set()):
                out["neighbors"].add(f"category:{c}")
            for imp in f.get("imports", set()):
                out["neighbors"].add(f"file:{imp}")
    elif entity.startswith("token:"):
        tok = entity[len("token:"):]
        if tok in graph.tokens:
            for c in graph.tokens[tok].get("defined_by", set()):
                out["neighbors"].add(f"category:{c}")

    return {k: sorted(v) for k, v in out.items()}