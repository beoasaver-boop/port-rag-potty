"""RAG portable · chunking semántico por tipo de archivo (genérico).

Cada archivo se trocea en "unidades significativas":
  · CSS      → una regla top-level = un chunk (selector + body)
  · JS/TS    → función / IIFE / clase / const-block = un chunk
  · Python   → función / clase / header / bloques top-level
  · MD       → sección por heading (H2/H3) + cabecera, con metodología
  · HTML     → bloques entre comentarios + secciones estructurales grandes
  · JSON     → top-level keys (objeto) o el array entero
  · YAML     → top-level keys; TOML → tablas [sección]
  · resto    → chunks por líneas (raw)

El dispatch ext → chunker lo decide `config.CHUNKERS` (editable en
rag.config.json). Cualquier extensión sin chunker cae en `raw`, así que el RAG
indexa cualquier proyecto sin configuración manual previa.

Cada chunk lleva metadata (kind, category, tokens, methodology, líneas, etc.)
para que el grafo y los filtros funcionen.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from . import config


# ── Modelo de chunk ─────────────────────────────────────────────────────────
@dataclass
class Chunk:
    id: str                  # hash estable: file + start_line + hash contenido
    text: str                # texto embebible (con heading de contexto)
    file: str                # ruta relativa
    ext: str                 # ".css", ".js"...
    kind: str                # css-rule | js-fn | py-def | md-section | ...
    category: str = ""       # categoría del grafo si aplica (era "theme")
    section: str = ""        # nombre legible: nombre función, heading, etc.
    start_line: int = 0
    end_line: int = 0
    tokens: list[str] = field(default_factory=list)
    methodology: bool = False

    def to_meta(self) -> dict:
        return {
            "file": self.file,
            "ext": self.ext,
            "kind": self.kind,
            "category": self.category,
            "section": self.section[:200],
            "start_line": self.start_line,
            "end_line": self.end_line,
            "tokens": ",".join(self.tokens[:32]),
            "methodology": self.methodology,
        }

    def compose_embed_text(self) -> str:
        """Texto completo que se enviaría al modelo de embeddings, SIN truncar."""
        parts = []
        if self.category:
            parts.append(f"[category={self.category}]")
        if self.section:
            parts.append(f"[{self.kind}:{self.section}]")
        parts.append(f"// {self.file}:{self.start_line}")
        parts.append(self.text)
        return "\n".join(parts)

    def embed_text(self) -> str:
        """Texto que se envía a bge-m3 (incluye contexto para mejorar retrieval).

        Último recurso: si pese a _enforce_embed_limit() sigue habiendo exceso,
        se trunca. Nunca debe llegar aquí un chunk partible.
        """
        out = self.compose_embed_text()
        max_chars = getattr(config, "EMBED_TEXT_MAX_CHARS", 4000)
        if len(out) > max_chars:
            out = out[: max_chars - 80] + "\n… (truncado por límite de embedding)"
        return out


# ── Utilidades ──────────────────────────────────────────────────────────────
def _stable_id(file: str, start: int, text: str) -> str:
    import hashlib
    h = hashlib.sha1(f"{file}|{start}|{text}".encode("utf-8")).hexdigest()[:12]
    rel = file.replace("/", "_").replace(".", "_")
    return f"{rel}_{start}_{h}"


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _categories_in_text(text: str, limit: int = 64) -> list[str]:
    if not config.CATEGORY_REGEX:
        return []
    return list({m for m in re.findall(config.CATEGORY_REGEX, text) if m})[:limit]


def _tokens_in_text(text: str, limit: int = 64) -> list[str]:
    if not config.TOKEN_REGEX:
        return []
    return list({m for m in re.findall(config.TOKEN_REGEX, text) if m})[:limit]


# ── Splitter genérico por longitud ──────────────────────────────────────────
def _split_long(text: str, max_chars: int, overlap: int) -> list[tuple[str, int, int]]:
    """Parte un texto largo en ventanas (texto, start_line, end_line)."""
    if len(text) <= max_chars:
        return [(text, 1, text.count("\n") + 1)]

    parts: list[tuple[str, int, int]] = []
    step = max(1, max_chars - overlap)
    pos = 0
    while pos < len(text):
        end = min(len(text), pos + max_chars)
        if end < len(text):
            cut = text.rfind("\n", pos, end)
            if cut > pos + 200:
                end = cut
        segment = text[pos:end]
        parts.append((segment, _line_of(text, pos), _line_of(text, end)))
        if end >= len(text):
            break
        pos = max(pos + step, end - overlap)
    return parts


# ── CSS ─────────────────────────────────────────────────────────────────────
_CSS_RULE = re.compile(
    r"(?P<sel>(?:[^{}/]|/(?!\*)[^/])+?)\{(?P<body>[^{}]*)\}",
    re.DOTALL,
)


def _strip_css_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def chunk_css(content: str, file_rel: str) -> list[Chunk]:
    no_comments = _strip_css_comments(content)
    chunks: list[Chunk] = []
    pos_in_orig = 0
    orig_text = content

    for m in _CSS_RULE.finditer(no_comments):
        selector = m.group("sel").strip()
        body = m.group("body").strip()
        if not selector or not body:
            continue

        sel_in_orig = orig_text.find(selector, pos_in_orig)
        if sel_in_orig < 0:
            continue
        pos_in_orig = sel_in_orig + len(selector)
        start_line = _line_of(orig_text, sel_in_orig)
        end_line = start_line + body.count("\n")

        max_chars = config.CHUNK_MAX_CHARS.get("css", 1800)
        target_chars = config.CHUNK_TARGET_CHARS.get("css", 900)

        body_chunks = _split_long(body, max_chars, config.CHUNK_OVERLAP)
        categories = _categories_in_text(selector, limit=1)
        category = categories[0] if categories else ""
        toks = _tokens_in_text(body)

        if len(body_chunks) == 1:
            text = f"{selector} {{\n  {body_chunks[0][0]}\n}}"
            chunks.append(Chunk(
                id=_stable_id(file_rel, start_line, text),
                text=text,
                file=file_rel, ext=".css", kind="css-rule",
                category=category,
                section=selector[:120],
                start_line=start_line,
                end_line=end_line,
                tokens=toks,
            ))
        else:
            for idx, (seg, sl, el) in enumerate(body_chunks):
                text = f"{selector} {{\n  {seg}\n}}"
                chunks.append(Chunk(
                    id=_stable_id(file_rel, start_line + sl - 1, text),
                    text=text,
                    file=file_rel, ext=".css", kind="css-rule",
                    category=category,
                    section=f"{selector[:80]} (parte {idx + 1})",
                    start_line=start_line + sl - 1,
                    end_line=start_line + el - 1,
                    tokens=_tokens_in_text(seg),
                ))

    # Comentarios CSS sueltos (sin regla): chunks "raw"
    for m in re.finditer(r"/\*(.*?)\*/", orig_text, flags=re.DOTALL):
        comment = m.group(1).strip()
        if len(comment) < 30:
            continue
        sl = _line_of(orig_text, m.start())
        text = f"/* (comentario CSS)\n{comment}\n*/"
        chunks.append(Chunk(
            id=_stable_id(file_rel, sl, text),
            text=text,
            file=file_rel, ext=".css", kind="css-comment",
            section="comentario",
            start_line=sl,
            end_line=_line_of(orig_text, m.end()),
        ))

    return chunks


# ── JS / TS ─────────────────────────────────────────────────────────────────
_JS_FUNCTION = re.compile(
    r"(?:^|\n)\s*(?:function\s+(?P<name1>\w+)|"
    r"(?:const|let|var)\s+(?P<name2>\w+)\s*=\s*(?:function|\([^)]*\)\s*=>|async\s+))",
)
_JS_CLASS = re.compile(r"(?:^|\n)\s*class\s+(?P<name>\w+)")
_JS_IIFE = re.compile(r"\(\s*function\s*\([^)]*\)\s*\{")


def _slice_balanced(text: str, open_idx: int) -> tuple[int, int]:
    """Devuelve (start, end) del bloque {...} balanceado que abre en open_idx."""
    depth = 0
    i = open_idx
    in_str = None
    in_tmpl = False
    while i < len(text):
        c = text[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == in_str:
                in_str = None
        elif in_tmpl:
            if c == "`":
                in_tmpl = False
        else:
            if c in ("'", '"'):
                in_str = c
            elif c == "`":
                in_tmpl = True
            elif c == "/" and i + 1 < len(text) and text[i + 1] == "/":
                nl = text.find("\n", i)
                i = nl + 1 if nl != -1 else len(text)
                continue
            elif c == "/" and i + 1 < len(text) and text[i + 1] == "*":
                end_c = text.find("*/", i + 2)
                i = end_c + 2 if end_c != -1 else len(text)
                continue
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return open_idx, i + 1
        i += 1
    return open_idx, len(text)


def _js_residual(content: str, consumed: list[tuple[int, int]], file_rel: str) -> list[Chunk]:
    """Código top-level NO cubierto por símbolos (huecos entre rangos consumidos)."""
    bounds = [0, len(content)]
    for a, b in consumed:
        bounds += [a, b]
    bounds = sorted(set(bounds))
    out: list[Chunk] = []

    def in_consumed(p: int) -> bool:
        return any(a <= p < b for a, b in consumed)

    for i in range(len(bounds) - 1):
        seg_start = bounds[i]
        seg_end = bounds[i + 1]
        if in_consumed(seg_start):
            continue
        seg = content[seg_start:seg_end].strip()
        if len(seg) < 80:
            continue
        sl = _line_of(content, seg_start)
        out.append(Chunk(
            id=_stable_id(file_rel, sl, seg[:200]),
            text=seg,
            file=file_rel, ext=Path(file_rel).suffix, kind="js-block",
            section="código top-level",
            start_line=sl,
            end_line=_line_of(content, seg_end),
        ))
    return out


def chunk_js(content: str, file_rel: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    consumed: list[tuple[int, int]] = []

    def in_consumed(p: int) -> bool:
        return any(a <= p < b for a, b in consumed)

    for m in _JS_FUNCTION.finditer(content):
        name = m.group("name1") or m.group("name2") or "anon"
        body_start = content.find("{", m.end())
        arrow_body_start = content.find("=>", m.end())
        ss = -1
        if body_start != -1 and (arrow_body_start == -1 or body_start < arrow_body_start):
            ss = body_start
        elif arrow_body_start != -1:
            semi = content.find(";", arrow_body_start)
            if semi != -1:
                txt = content[m.start():semi + 1]
                sl = _line_of(content, m.start())
                chunks.append(Chunk(
                    id=_stable_id(file_rel, sl, txt),
                    text=txt,
                    file=file_rel, ext=Path(file_rel).suffix, kind="js-arrow",
                    section=name,
                    start_line=sl,
                    end_line=_line_of(content, semi),
                ))
                consumed.append((m.start(), semi + 1))
                continue
        if ss == -1:
            continue
        if in_consumed(m.start()):
            continue
        a, b = _slice_balanced(content, ss)
        txt = content[m.start():b]
        sl = _line_of(content, m.start())
        chunks.append(Chunk(
            id=_stable_id(file_rel, sl, txt),
            text=txt,
            file=file_rel, ext=Path(file_rel).suffix, kind="js-fn",
            section=name,
            start_line=sl,
            end_line=_line_of(content, b),
        ))
        consumed.append((m.start(), b))

    for m in _JS_CLASS.finditer(content):
        if in_consumed(m.start()):
            continue
        name = m.group("name")
        brace = content.find("{", m.end())
        if brace == -1:
            continue
        a, b = _slice_balanced(content, brace)
        txt = content[m.start():b]
        sl = _line_of(content, m.start())
        chunks.append(Chunk(
            id=_stable_id(file_rel, sl, txt),
            text=txt,
            file=file_rel, ext=Path(file_rel).suffix, kind="js-class",
            section=name,
            start_line=sl,
            end_line=_line_of(content, b),
        ))
        consumed.append((m.start(), b))

    for m in _JS_IIFE.finditer(content):
        if in_consumed(m.start()):
            continue
        brace = content.find("{", m.end())
        if brace == -1:
            continue
        a, b = _slice_balanced(content, brace)
        txt = content[m.start():b]
        sl = _line_of(content, m.start())
        chunks.append(Chunk(
            id=_stable_id(file_rel, sl, txt),
            text=txt,
            file=file_rel, ext=Path(file_rel).suffix, kind="js-iife",
            section="IIFE",
            start_line=sl,
            end_line=_line_of(content, b),
        ))
        consumed.append((m.start(), b))

    # Cabecera del archivo (comentarios/requires hasta el primer símbolo)
    header_end = 0
    for m in re.finditer(r"^(?!\s*$)(?!\s*//)(?!\s*/\*)", content, re.MULTILINE):
        if m.start() > 0:
            header_end = m.start()
            break
    if header_end > 0:
        head = content[:header_end].strip()
        if head and len(head) >= 30:
            chunks.append(Chunk(
                id=_stable_id(file_rel, 1, head),
                text=head,
                file=file_rel, ext=Path(file_rel).suffix, kind="js-header",
                section="cabecera",
                start_line=1,
                end_line=_line_of(content, header_end),
            ))
            consumed.append((0, header_end))

    chunks.extend(_js_residual(content, consumed, file_rel))
    return chunks


# ── Python ──────────────────────────────────────────────────────────────────
_PY_DEF = re.compile(r"^(?:(?:async\s+)?(?:def|class)\s+)(?P<name>\w+)", re.MULTILINE)


def _py_indent(content: str, pos: int) -> int:
    """Indentación (en espacios) de la línea que contiene pos."""
    line_start = content.rfind("\n", 0, pos) + 1
    n = 0
    for c in content[line_start:pos]:
        if c == " ":
            n += 1
        elif c == "\t":
            n += 4
        elif c == "\r":
            continue
        else:
            break
    return n


def _py_dec_start(content: str, def_start: int) -> int:
    """Inicio del run de decoradores que precede a def_start (o el propio start)."""
    s = def_start
    while s > 0:
        nl = content.rfind("\n", 0, s)
        line_start = nl + 1
        line = content[line_start:s].strip()
        if not line.startswith("@"):
            break
        s = line_start
    return s


def chunk_py(content: str, file_rel: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    matches = list(_PY_DEF.finditer(content))
    consumed: list[tuple[int, int]] = []

    def in_consumed(p: int) -> bool:
        return any(a <= p < b for a, b in consumed)

    for i, m in enumerate(matches):
        if in_consumed(m.start()):
            continue
        base_indent = _py_indent(content, m.start())
        # El bloque termina en el siguiente symbol con indent <= base_indent
        end = len(content)
        for j in range(i + 1, len(matches)):
            nxt = matches[j]
            if _py_indent(content, nxt.start()) <= base_indent:
                end = _py_dec_start(content, nxt.start())
                break
        start = _py_dec_start(content, m.start())
        txt = content[start:end].strip()
        if not txt:
            continue
        kind = "py-class" if m.group(0).lstrip().startswith("class") else "py-def"
        sl = _line_of(content, start)
        chunks.append(Chunk(
            id=_stable_id(file_rel, sl, txt[:200]),
            text=txt,
            file=file_rel, ext=".py", kind=kind,
            section=m.group("name"),
            start_line=sl,
            end_line=_line_of(content, end),
        ))
        consumed.append((start, end))

    # Cabecera del módulo (docstring + comentarios hasta el primer símbolo)
    header_end = matches[0].start() if matches else len(content)
    head = content[:header_end].strip()
    if head and len(head) >= 30:
        chunks.append(Chunk(
            id=_stable_id(file_rel, 1, head[:200]),
            text=head,
            file=file_rel, ext=".py", kind="py-header",
            section="módulo",
            start_line=1,
            end_line=_line_of(content, header_end),
        ))
        consumed.append((0, header_end))

    # Residual top-level: imports, constantes, bloques if __name__ …
    chunks.extend(_js_residual(content, consumed, file_rel))
    return chunks


# ── Markdown ────────────────────────────────────────────────────────────────
_MD_SPLIT = re.compile(r"(?m)^(#{2,4})\s+(.+?)\s*$")


def chunk_md(content: str, file_rel: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    headings = list(_MD_SPLIT.finditer(content))
    if not headings:
        chunks.append(Chunk(
            id=_stable_id(file_rel, 1, content),
            text=content,
            file=file_rel, ext=".md", kind="md-section",
            section="documento completo",
            start_line=1,
            end_line=content.count("\n") + 1,
            methodology=config.is_methodology_file(file_rel),
        ))
        return chunks

    # Cabecera: antes del primer H2 (título H1 + intro)
    first = headings[0]
    if first.start() > 0:
        head = content[:first.start()].strip()
        if head:
            chunks.append(Chunk(
                id=_stable_id(file_rel, 1, head),
                text=head,
                file=file_rel, ext=".md", kind="md-section",
                section="cabecera",
                start_line=1,
                end_line=_line_of(content, first.start()),
                methodology=config.is_methodology_file(file_rel),
            ))

    for i, h in enumerate(headings):
        start = h.start()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(content)
        body = content[start:end].strip()
        if len(body) < 30:
            continue
        sl = _line_of(content, start)
        max_chars = config.CHUNK_MAX_CHARS.get("md", 2000)
        heading_text = h.group(2)
        is_meth = config.is_methodology_heading(file_rel, heading_text, h.group(0))
        for j, (seg, ssl, eel) in enumerate(_split_long(body, max_chars, config.CHUNK_OVERLAP)):
            chunks.append(Chunk(
                id=_stable_id(file_rel, sl + ssl - 1, seg[:200]),
                text=seg,
                file=file_rel, ext=".md", kind="md-section",
                section=heading_text[:120] + (f" (parte {j + 1})" if j > 0 else ""),
                start_line=sl + ssl - 1,
                end_line=sl + eel - 1,
                methodology=is_meth,
            ))

    return chunks


# ── HTML ────────────────────────────────────────────────────────────────────
_HTML_COMMENT = re.compile(r"<!--(.*?)-->", re.DOTALL)


def chunk_html(content: str, file_rel: str) -> list[Chunk]:
    chunks: list[Chunk] = []

    for m in _HTML_COMMENT.finditer(content):
        text = m.group(0).strip()
        if len(text) < 40:
            continue
        sl = _line_of(content, m.start())
        chunks.append(Chunk(
            id=_stable_id(file_rel, sl, text),
            text=text,
            file=file_rel, ext=".html", kind="html-block",
            section="comentario",
            start_line=sl,
            end_line=_line_of(content, m.end()),
        ))

    for m in re.finditer(r"<(div|main|section|aside|nav|header|footer|form|script|style)\b[^>]*>", content):
        tag = m.group(1)
        close = re.search(rf"</{tag}\s*>", content[m.end():])
        if not close:
            continue
        block_start = m.start()
        block_end = m.end() + close.end()
        block = content[block_start:block_end]
        if len(block) < 200:
            continue
        attr_match = re.search(r'class\s*=\s*["\']([\w\- ]+)["\']', m.group(0))
        section = attr_match.group(1) if attr_match else tag
        sl = _line_of(content, block_start)
        chunks.append(Chunk(
            id=_stable_id(file_rel, sl, block[:200]),
            text=block,
            file=file_rel, ext=".html", kind="html-block",
            section=section,
            start_line=sl,
            end_line=_line_of(content, block_end),
        ))

    return chunks


# ── JSON ────────────────────────────────────────────────────────────────────
def chunk_json(content: str, file_rel: str) -> list[Chunk]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return [Chunk(
            id=_stable_id(file_rel, 1, content),
            text=content,
            file=file_rel, ext=".json", kind="json",
            section="raw",
            start_line=1,
            end_line=content.count("\n") + 1,
        )]
    chunks: list[Chunk] = []
    if isinstance(data, dict):
        for key, val in data.items():
            text = json.dumps({key: val}, indent=2, ensure_ascii=False)
            chunks.append(Chunk(
                id=_stable_id(file_rel, 1, key + text[:100]),
                text=text,
                file=file_rel, ext=".json", kind="json",
                section=str(key),
                start_line=1,
                end_line=text.count("\n") + 1,
            ))
    else:
        chunks.append(Chunk(
            id=_stable_id(file_rel, 1, content[:200]),
            text=content,
            file=file_rel, ext=".json", kind="json",
            section="root",
            start_line=1,
            end_line=content.count("\n") + 1,
        ))
    return chunks


# ── YAML / TOML / raw ───────────────────────────────────────────────────────
def _slice_sections(content: str, boundaries: list[int], file_rel: str,
                    ext: str, kind: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for i in range(len(boundaries) - 1):
        seg_start = boundaries[i]
        seg_end = boundaries[i + 1]
        seg = content[seg_start:seg_end].strip()
        if len(seg) < 30:
            continue
        heading = None
        if kind == "yaml-block":
            m = re.match(r"^([A-Za-z0-9_.-]+)\s*:", seg)
            heading = m.group(1) if m else None
        elif kind == "toml-section":
            m = re.match(r"^\[\s*([\w.\-]+)\s*\]", seg)
            heading = m.group(1) if m else None
        sl = _line_of(content, seg_start)
        chunks.append(Chunk(
            id=_stable_id(file_rel, sl, seg[:200]),
            text=seg,
            file=file_rel, ext=ext, kind=kind,
            section=heading or f"sección {i}",
            start_line=sl,
            end_line=_line_of(content, seg_end),
        ))
    return chunks


_YAML_KEY = re.compile(r"^[A-Za-z0-9_.-]+\s*:(?:\s|$)", re.MULTILINE)
_TOML_TABLE = re.compile(r"^\[\s*[\w.\-]+\s*\]\s*$", re.MULTILINE)


def chunk_yaml(content: str, file_rel: str) -> list[Chunk]:
    keys = list(_YAML_KEY.finditer(content))
    if not keys:
        return _slice_sections(content, [0, len(content)], file_rel, ".yaml", "yaml-block")
    boundaries = [0] + [m.start() for m in keys] + [len(content)]
    return _slice_sections(content, boundaries, file_rel, ".yaml", "yaml-block")


def chunk_toml(content: str, file_rel: str) -> list[Chunk]:
    tables = list(_TOML_TABLE.finditer(content))
    if not tables:
        return _slice_sections(content, [0, len(content)], file_rel, ".toml", "toml-section")
    boundaries = [0] + [m.start() for m in tables] + [len(content)]
    return _slice_sections(content, boundaries, file_rel, ".toml", "toml-section")


def chunk_raw(content: str, file_rel: str) -> list[Chunk]:
    max_chars = config.CHUNK_MAX_CHARS.get("raw", 3000)
    parts = _split_long(content, max_chars, config.CHUNK_OVERLAP)
    chunks: list[Chunk] = []
    for idx, (seg, sl, el) in enumerate(parts):
        chunks.append(Chunk(
            id=_stable_id(file_rel, sl, seg[:200]),
            text=seg,
            file=file_rel, ext=Path(file_rel).suffix, kind="raw",
            section=f"texto ({idx + 1}/{len(parts)})" if len(parts) > 1 else "texto",
            start_line=sl,
            end_line=el,
        ))
    return chunks


# ── Enforcement del presupuesto de embedding ───────────────────────────────
def _enforce_embed_limit(chunks: list[Chunk], file_rel: str) -> list[Chunk]:
    """Garantiza que ningún chunk supere el presupuesto de embedding.

    bge-m3 en Ollama rechaza con HTTP 500 ("the input length exceeds the context
    length") los textos demasiado largos, así que truncar en embed_text() no
    basta: partimos el chunk en sub-chunks con id estable propio y ajustamos
    las líneas. Nunca debe salir de aquí un chunk que Ollama vaya a rechazar.
    """
    max_chars = getattr(config, "EMBED_TEXT_MAX_CHARS", 4000)
    budget = max(400, max_chars - 200)  # margen para el prefijo de contexto
    out: list[Chunk] = []
    for ch in chunks:
        if len(ch.compose_embed_text()) <= max_chars:
            out.append(ch)
            continue
        parts = _split_long(ch.text, budget, config.CHUNK_OVERLAP)
        if len(parts) == 1 and parts[0][0] == ch.text:
            out.append(ch)
            continue
        n = len(parts)
        for idx, (segment, p_start, p_end) in enumerate(parts, start=1):
            start = ch.start_line + p_start - 1
            out.append(Chunk(
                id=_stable_id(file_rel, start, segment[:200]),
                text=segment,
                file=ch.file,
                ext=ch.ext,
                kind=ch.kind,
                category=ch.category,
                section=(f"{ch.section} [{idx}/{n}]" if ch.section else f"[{idx}/{n}]")[:200],
                start_line=start,
                end_line=ch.start_line + p_end - 1,
                tokens=ch.tokens,
                methodology=ch.methodology,
            ))
    return out


# ── Dispatcher ──────────────────────────────────────────────────────────────
_CHUNKER_BY_KEY: dict[str, Callable[[str, str], list[Chunk]]] = {
    "css": chunk_css,
    "js": chunk_js,
    "md": chunk_md,
    "html": chunk_html,
    "json": chunk_json,
    "py": chunk_py,
    "yaml": chunk_yaml,
    "toml": chunk_toml,
    "raw": chunk_raw,
}


def chunk_file(path: Path, project_root: Path) -> list[Chunk]:
    try:
        content = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    rel = str(path.relative_to(project_root))
    ext = path.suffix.lower()
    key = config.CHUNKERS.get(ext, "raw")
    fn = _CHUNKER_BY_KEY.get(key, chunk_raw)
    chunks = fn(content, rel)
    return _enforce_embed_limit(chunks, rel)


def iter_project_files(project_root: Path) -> Iterable[Path]:
    for glob in config.INDEX_GLOBS:
        for p in project_root.glob(glob):
            if not p.is_file():
                continue
            rel = p.relative_to(project_root)
            parts = rel.parts
            if any(part in config.EXCLUDE_DIRS for part in parts):
                continue
            if p.name in config.EXCLUDE_FILES:
                continue
            yield p