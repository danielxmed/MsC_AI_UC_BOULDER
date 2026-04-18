#!/usr/bin/env python3
"""Build the interactive web version of the EPUB ebook.

Extracts content from the canonical EPUB (`deep_learning_modulo_2_ebook.epub`)
and produces a self-contained SPA under `interactive_ebook/`, along with a
PWA manifest, service worker and GitHub Pages deployment artifacts.

Usage:
    python3 build_ebook/build_interactive.py

No CLI arguments: paths are resolved relative to the repo root.
"""
from __future__ import annotations

import hashlib
import html as html_module
import io
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
EPUB_PATH = REPO_ROOT / "deep_learning_modulo_2_ebook.epub"
OUT_DIR = REPO_ROOT / "interactive_ebook"
MEDIA_OUT = OUT_DIR / "media"

# Parts detected from nav.xhtml (by title keyword). Chapters inherit the
# most recent part encountered in TOC order.
PART_PATTERNS = [
    (re.compile(r"Parte I\b"), "parte-i"),
    (re.compile(r"Parte II\b"), "parte-ii"),
    (re.compile(r"Parte III\b"), "parte-iii"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def slurp_epub(path: Path) -> dict[str, bytes]:
    """Return {zip_name: bytes} for every entry in the EPUB."""
    contents: dict[str, bytes] = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            contents[name] = z.read(name)
    return contents


def parse_nav(nav_bytes: bytes) -> list[dict]:
    """Parse EPUB/nav.xhtml → flat list of TOC entries.

    Each entry: {id, title, href (chNNN.xhtml#anchor or chNNN.xhtml),
                 chapter_file, anchor, level (1=chapter, 2=section),
                 part_slug, part_title}.
    """
    soup = BeautifulSoup(nav_bytes, "html.parser")
    toc_root = soup.find("nav", attrs={"epub:type": "toc"}) or soup.find("nav")
    entries: list[dict] = []
    current_part_slug = "frontmatter"
    current_part_title = "Início"

    def walk(ol: Tag, depth: int = 1, parent_chapter: str | None = None):
        nonlocal current_part_slug, current_part_title
        for li in ol.find_all("li", recursive=False):
            a = li.find("a", recursive=False)
            if not a:
                continue
            href = (a.get("href") or "").strip()
            title = a.get_text(strip=True)

            # Detect Part dividers by title
            for pat, slug in PART_PATTERNS:
                if pat.search(title):
                    current_part_slug = slug
                    current_part_title = title
                    break

            chapter_file, _, anchor = href.partition("#")
            chapter_file = chapter_file.replace("text/", "")

            entry = {
                "title": title,
                "href": href,
                "chapter_file": chapter_file,
                "anchor": anchor or None,
                "level": depth,
                "part_slug": current_part_slug,
                "part_title": current_part_title,
                "parent_chapter": parent_chapter,
            }
            entries.append(entry)

            nested = li.find("ol", recursive=False)
            if nested:
                walk(nested, depth + 1, parent_chapter=chapter_file)

    root_ol = toc_root.find("ol") if toc_root else None
    if root_ol:
        walk(root_ol)
    return entries


def slugify_chapter_id(file_name: str) -> str:
    m = re.match(r"ch(\d+)\.xhtml", file_name)
    return f"ch{m.group(1)}" if m else file_name.replace(".xhtml", "")


def is_update_box_section(el: Tag) -> bool:
    return (
        isinstance(el, Tag)
        and el.name == "section"
        and "update-box" in (el.get("class") or [])
    )


def transform_update_boxes(soup: BeautifulSoup) -> None:
    """Convert `<section class="... update-box">` → `<details class="update-box" open>`.

    The first heading inside becomes the `<summary>`. Preserves all child
    content in the same order.
    """
    for section in list(soup.find_all("section")):
        if not is_update_box_section(section):
            continue
        details = soup.new_tag("details")
        details["class"] = ["update-box"]
        details["open"] = ""

        # Preserve id/other attrs of the original section
        if section.get("id"):
            details["id"] = section["id"]

        # Find first heading (h2/h3/h4) and promote to summary
        heading = section.find(["h2", "h3", "h4", "h5", "h6"])
        summary = soup.new_tag("summary")
        if heading:
            # Move heading content into summary as-is, keep the original heading tag
            # inside summary so visual hierarchy survives but disclosure triangle works.
            heading.extract()
            summary.append(heading)
        else:
            summary.string = "Atualização editorial"
        details.append(summary)

        # Move remaining children into details
        for child in list(section.children):
            details.append(child.extract() if isinstance(child, Tag) else child)

        section.replace_with(details)


STATUS_MAP = {
    "🟢": "status-current",
    "🟡": "status-partial",
    "🔴": "status-legacy",
}

STATUS_LABELS = {
    "🟢": "Atual",
    "🟡": "Parcial",
    "🔴": "Legado",
}

STATUS_RE = re.compile(r"[🟢🟡🔴]")


def wrap_status_badges(soup: BeautifulSoup) -> None:
    """Wrap lone 🟢/🟡/🔴 glyphs in `<span class="status-badge status-...">`.

    Walk text nodes; whenever a badge character appears, split the text and
    insert a `<span>`. Avoids double-wrapping inside existing badges.
    """
    def walk(parent: Tag):
        for child in list(parent.children):
            if isinstance(child, NavigableString):
                text = str(child)
                if not STATUS_RE.search(text):
                    continue
                # Skip if already inside a status-badge
                if any(
                    isinstance(a, Tag) and "status-badge" in (a.get("class") or [])
                    for a in child.parents
                ):
                    continue
                # Build replacement sequence
                parts = []
                last = 0
                for m in STATUS_RE.finditer(text):
                    if m.start() > last:
                        parts.append(NavigableString(text[last:m.start()]))
                    glyph = m.group(0)
                    span = soup.new_tag("span")
                    span["class"] = ["status-badge", STATUS_MAP[glyph]]
                    span["title"] = STATUS_LABELS[glyph]
                    span["aria-label"] = STATUS_LABELS[glyph]
                    span.string = glyph
                    parts.append(span)
                    last = m.end()
                if last < len(text):
                    parts.append(NavigableString(text[last:]))
                # Replace
                for p in parts:
                    child.insert_before(p)
                child.extract()
            elif isinstance(child, Tag):
                walk(child)

    body = soup.body or soup
    walk(body)


def add_code_language(soup: BeautifulSoup) -> None:
    """Mark `<pre>` with `data-lang` derived from the inner `<code class="...">`."""
    for pre in soup.find_all("pre"):
        code = pre.find("code")
        if not code:
            continue
        classes = code.get("class") or []
        lang = None
        for c in classes:
            if c.startswith("language-"):
                lang = c[len("language-"):]
                break
            if c.startswith("sourceCode") or c == "sourceCode":
                continue
        # pandoc often emits: class="sourceCode python"
        if not lang:
            for c in classes:
                if c not in ("sourceCode",) and not c.startswith("language-"):
                    lang = c
                    break
        if lang:
            pre["data-lang"] = lang


def rewrite_image_paths(soup: BeautifulSoup, lazy_for_chapter: bool) -> None:
    for img in soup.find_all("img"):
        src = img.get("src", "")
        src = src.replace("../media/", "media/").replace("media/", "media/")
        img["src"] = src
        if lazy_for_chapter:
            img["loading"] = "lazy"
            img["decoding"] = "async"
        else:
            # keep first chapter eager for immediate paint
            img["decoding"] = "async"


def extract_chapter_body(raw_xhtml: bytes, chapter_id: str, part_slug: str,
                         lazy: bool) -> str:
    """Return the processed `<section>…</section>` HTML string for a chapter."""
    soup = BeautifulSoup(raw_xhtml, "html.parser")
    body = soup.body
    if body is None:
        return ""

    # Remove `<link rel="stylesheet">` or `<style>` that pandoc inlined inside body (none)
    # They live in <head>, which we already dropped by taking body.

    transform_update_boxes(soup)
    rewrite_image_paths(soup, lazy_for_chapter=lazy)
    add_code_language(soup)
    wrap_status_badges(soup)

    # Unwrap the top-level body into a single wrapper <section>
    wrapper = soup.new_tag("section")
    wrapper["class"] = ["chapter"]
    wrapper["data-chapter-id"] = chapter_id
    wrapper["data-part"] = part_slug
    wrapper["id"] = chapter_id
    for child in list(body.children):
        wrapper.append(child.extract() if isinstance(child, Tag) else child)

    return str(wrapper)


def hoist_existing_ids(section_html: str) -> str:
    """No-op placeholder (IDs survive BS4 round-trip)."""
    return section_html


# ---------------------------------------------------------------------------
# Icon + manifest + service worker generation
# ---------------------------------------------------------------------------

def generate_icons(cover_bytes: bytes, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cover = Image.open(io.BytesIO(cover_bytes)).convert("RGB")
    sizes = {
        "apple-touch-icon.png": 180,
        "icon-192.png": 192,
        "icon-512.png": 512,
    }
    generated: dict[str, Path] = {}
    for name, size in sizes.items():
        # Center-crop then resize
        w, h = cover.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        cropped = cover.crop((left, top, left + side, top + side))
        resized = cropped.resize((size, size), Image.LANCZOS)
        dest = out_dir / name
        resized.save(dest, "PNG", optimize=True)
        generated[name] = dest
    return generated


def write_manifest(path: Path) -> None:
    manifest = {
        "name": "Otimização em Deep Learning",
        "short_name": "DL Optimization",
        "description": "Ebook interativo do Módulo 2 do MsC AI - UC Boulder.",
        "lang": "pt-BR",
        "start_url": "./",
        "scope": "./",
        "display": "standalone",
        "orientation": "any",
        "theme_color": "#1a3a5c",
        "background_color": "#fdf8f0",
        "icons": [
            {"src": "media/icon-192.png", "sizes": "192x192", "type": "image/png"},
            {"src": "media/icon-512.png", "sizes": "512x512", "type": "image/png"},
            {
                "src": "media/icon-512.png",
                "sizes": "512x512",
                "type": "image/png",
                "purpose": "maskable",
            },
        ],
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def write_service_worker(path: Path, cache_version: str, precache: list[str]) -> None:
    precache_json = json.dumps(precache, indent=2)
    js = f"""'use strict';
const CACHE = 'dl-opt-{cache_version}';
const PRECACHE_URLS = {precache_json};

self.addEventListener('install', (event) => {{
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting())
  );
}});

self.addEventListener('activate', (event) => {{
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(
      keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))
    )).then(() => self.clients.claim())
  );
}});

self.addEventListener('fetch', (event) => {{
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  event.respondWith(
    caches.match(req).then((cached) => {{
      if (cached) return cached;
      return fetch(req).then((resp) => {{
        if (!resp || resp.status !== 200 || resp.type !== 'basic') return resp;
        const copy = resp.clone();
        caches.open(CACHE).then((cache) => cache.put(req, copy));
        return resp;
      }}).catch(() => {{
        if (req.mode === 'navigate') return caches.match('./index.html');
      }});
    }})
  );
}});
"""
    path.write_text(js, encoding="utf-8")


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

def build_toc_html(entries: list[dict]) -> str:
    """Build the sidebar TOC HTML from flat TOC entries."""
    lines = ['<ol class="toc-root" role="list">']
    stack_depth = 1
    for i, e in enumerate(entries):
        depth = e["level"]
        while stack_depth < depth:
            lines.append('<ol role="list">')
            stack_depth += 1
        while stack_depth > depth:
            lines.append("</ol></li>")
            stack_depth -= 1
        # Build link. Root-level entries get chapter navigation.
        chapter_id = slugify_chapter_id(e["chapter_file"])
        href = f"#{e['anchor']}" if e["anchor"] else f"#{chapter_id}"
        safe_title = html_module.escape(e["title"])
        is_part = any(pat.search(e["title"]) for pat, _ in PART_PATTERNS)
        classes = ["toc-entry", f"toc-depth-{depth}"]
        if is_part:
            classes.append("toc-part")
        classes.append(f"toc-part-{e['part_slug']}")
        cls = " ".join(classes)
        data_target = chapter_id if not e["anchor"] else e["anchor"]
        # Stay open after clicking (SPA manages scroll)
        lines.append(
            f'<li class="{cls}">'
            f'<a href="{href}" data-toc-target="{data_target}" data-chapter="{chapter_id}">'
            f'<span class="toc-title">{safe_title}</span>'
            f'<span class="toc-check" aria-hidden="true">✓</span>'
            f'</a>'
        )
        # Determine if next entry has deeper level (will open <ol>) or same/shallower (close <li>)
        next_depth = entries[i + 1]["level"] if i + 1 < len(entries) else 0
        if next_depth <= depth:
            lines.append("</li>")
    while stack_depth > 1:
        lines.append("</ol></li>")
        stack_depth -= 1
    lines.append("</ol>")
    return "\n".join(lines)


def load_css() -> str:
    return CSS_TEMPLATE


def load_js() -> str:
    return JS_TEMPLATE


def build_index_html(toc_html: str, chapters_html: str, css: str, js: str) -> str:
    return HTML_TEMPLATE.format(
        css=css,
        js=js,
        toc=toc_html,
        chapters=chapters_html,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build() -> None:
    print(f"[build] reading EPUB: {EPUB_PATH}")
    if not EPUB_PATH.exists():
        sys.exit(f"EPUB not found at {EPUB_PATH}")

    data = slurp_epub(EPUB_PATH)

    # 1. Parse nav
    nav_bytes = data["EPUB/nav.xhtml"]
    toc_entries = parse_nav(nav_bytes)
    print(f"[build] TOC entries: {len(toc_entries)}")

    # 2. Derive chapter-to-part map from TOC (first occurrence)
    chapter_part: dict[str, tuple[str, str]] = {}
    for e in toc_entries:
        chapter_part.setdefault(
            e["chapter_file"], (e["part_slug"], e["part_title"])
        )

    # 3. Find chapter files in spine order
    chapter_files = sorted(
        [n for n in data if re.match(r"EPUB/text/ch\d+\.xhtml$", n)]
    )
    print(f"[build] chapter files: {len(chapter_files)}")

    # 4. Extract & transform each chapter
    chapter_sections: list[str] = []
    for idx, name in enumerate(chapter_files):
        chapter_file = name.replace("EPUB/text/", "")
        chapter_id = slugify_chapter_id(chapter_file)
        part_slug, _ = chapter_part.get(chapter_file, ("frontmatter", ""))
        lazy = idx > 0  # eager-decode first chapter only
        html = extract_chapter_body(data[name], chapter_id, part_slug, lazy)
        chapter_sections.append(html)
    chapters_html = "\n".join(chapter_sections)

    # 5. Prepare output dir & media
    if OUT_DIR.exists():
        # preserve any hand-written files (none expected); rebuild media fully
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    MEDIA_OUT.mkdir()

    # Copy all media files
    media_names = [n for n in data if n.startswith("EPUB/media/")]
    for n in media_names:
        base = Path(n).name
        (MEDIA_OUT / base).write_bytes(data[n])

    # Generate PWA icons from cover
    generate_icons(data["EPUB/media/cover.png"], MEDIA_OUT)

    # 6. Build TOC
    toc_html = build_toc_html(toc_entries)

    # 7. Build index.html
    html = build_index_html(toc_html, chapters_html, CSS_TEMPLATE, JS_TEMPLATE)

    # Hash for cache version before writing (content-addressed)
    cache_version = hashlib.sha1(html.encode("utf-8")).hexdigest()[:8]

    # Replace cache token
    html = html.replace("__CACHE_VERSION__", cache_version)
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")

    # 8. Manifest
    write_manifest(OUT_DIR / "manifest.webmanifest")

    # 9. Service worker — precache list includes all media + root shell
    media_entries = sorted(p.name for p in MEDIA_OUT.iterdir())
    precache = ["./", "./index.html", "./manifest.webmanifest"] + [
        f"./media/{m}" for m in media_entries
    ]
    write_service_worker(OUT_DIR / "sw.js", cache_version, precache)

    # 10. README
    write_readme(OUT_DIR / "README.md")

    # Summary
    html_size = (OUT_DIR / "index.html").stat().st_size
    print(f"[build] wrote {OUT_DIR / 'index.html'} ({html_size/1024:.1f} KB)")
    print(f"[build] media files: {len(list(MEDIA_OUT.iterdir()))}")
    print(f"[build] cache version: {cache_version}")


# ---------------------------------------------------------------------------
# README content
# ---------------------------------------------------------------------------

def write_readme(path: Path) -> None:
    text = """# Otimização em Deep Learning — versão interativa

Versão web interativa do ebook `deep_learning_modulo_2_ebook.epub`, com busca
client-side, dark mode, navegação por swipe (mobile), marcador de progresso,
lightbox de imagens e atalhos de teclado.

O conteúdo (texto, equações MathML e imagens) é extraído **exatamente** do
EPUB canônico — nenhuma palavra é perdida ou reescrita.

## Live

Após o primeiro push para `main`, o GitHub Pages publica automaticamente em:

    https://<user>.github.io/<repo>/

Para descobrir a URL exata:

```bash
gh repo view --json nameWithOwner -q .nameWithOwner
```

Setup único no GitHub: **Settings → Pages → Source: GitHub Actions**.

## Como ler no iPhone / iPad

1. Abra a URL acima no **Safari** (não Chrome — o *Add to Home Screen* do
   Chrome iOS é limitado).
2. Toque em **Compartilhar** → **Adicionar à Tela de Início**.
3. O ícone (derivado da capa) aparece na home. Tocar abre em modo
   *standalone* (sem barra de URL) e, após a primeira abertura online, o
   ebook funciona **offline para sempre** — avião, metrô, sem sinal.

## Como ler offline no desktop

```bash
git clone <repo>
cd <repo>/interactive_ebook
python3 -m http.server 8000
# abra http://localhost:8000
```

Abrir `index.html` diretamente com `file://` também funciona, porém o
service worker só registra em `http(s)://`.

## Features

- **Busca**: `/` foca a caixa de busca (filtra TOC + destaca ocorrências).
- **Dark mode**: tecla `d` ou o botão no topbar. Persistente via
  `localStorage`.
- **Navegação**:
  - Desktop: `j`/`k` scroll, `←`/`→` capítulo anterior/próximo, `t` TOC,
    `Esc` fecha busca/lightbox, `?` mostra legenda.
  - Mobile: swipe horizontal (threshold 80px) para capítulo anterior/próximo.
- **Lightbox**: clique/tap em qualquer imagem amplia em overlay. Fecha por
  `Esc`, tap fora ou ✕.
- **Código**: botão “copiar” em cada `<pre>`.
- **Marcador**: capítulos lidos ganham ✓ no TOC (via `localStorage`).
- **Filtros de status**: botões 🟢 / 🟡 / 🔴 no topbar filtram conteúdo
  por status (atual / parcial / legado).

## Como atualizar

O conteúdo é regenerado a partir do EPUB via:

```bash
python3 build_ebook/build_interactive.py
```

O workflow `.github/workflows/pages.yml` redeploy automaticamente em cada
push para `main` que toque `interactive_ebook/**`.
"""
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Templates — CSS, JS, HTML skeleton live here to keep build_interactive.py
# self-contained.
# ---------------------------------------------------------------------------

CSS_TEMPLATE = r"""
:root {
  --bg: #fdf8f0;
  --bg-soft: #f5eedd;
  --fg: #1c1a15;
  --fg-soft: #4a443a;
  --accent: #1a3a5c;
  --accent-soft: #3a7098;
  --accent-hover: #0f2c48;
  --violet: #5b3d8c;
  --violet-soft: #e9e2f4;
  --rule: #d9ceb6;
  --panel: #fffdf7;
  --code-bg: #f2ead8;
  --code-fg: #2a1e0a;
  --badge-current-bg: #d7ebd0;
  --badge-current-fg: #1e5c1f;
  --badge-partial-bg: #f7e7b5;
  --badge-partial-fg: #7a5a06;
  --badge-legacy-bg: #f4cfcf;
  --badge-legacy-fg: #6b1c1c;
  --update-bg: #eef2fb;
  --update-border: #4a6fa5;
  --shadow: 0 1px 2px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.05);
  --topbar-h: 56px;
  --sidebar-w: 320px;
  --font-serif: "Charter", "Iowan Old Style", "Palatino Linotype", "Georgia", "Noto Serif", serif;
  --font-sans: "Inter", "SF Pro Text", "Helvetica Neue", -apple-system, BlinkMacSystemFont, sans-serif;
  --font-mono: "Fira Code", "JetBrains Mono", "SF Mono", "Menlo", "Consolas", monospace;
  --transition: 200ms cubic-bezier(.2,.8,.2,1);
}

[data-theme="dark"] {
  --bg: #0f1419;
  --bg-soft: #161c23;
  --fg: #e8e4db;
  --fg-soft: #a9a196;
  --accent: #7fb5dc;
  --accent-soft: #5a9bcc;
  --accent-hover: #a8cdea;
  --violet: #b9a3e0;
  --violet-soft: #2a233a;
  --rule: #2e3541;
  --panel: #161c23;
  --code-bg: #1b2028;
  --code-fg: #e0dac7;
  --badge-current-bg: #1a3a1e;
  --badge-current-fg: #a9d6a3;
  --badge-partial-bg: #403516;
  --badge-partial-fg: #f0d68d;
  --badge-legacy-bg: #3e1f1f;
  --badge-legacy-fg: #efb2b2;
  --update-bg: #1a2232;
  --update-border: #7fb5dc;
  --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 4px 14px rgba(0,0,0,0.4);
}

* { box-sizing: border-box; }

html {
  font-size: clamp(16px, 1rem + 0.2vw, 18px);
  -webkit-text-size-adjust: 100%;
  scroll-behavior: smooth;
}

@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
  * { transition: none !important; animation: none !important; }
}

body {
  margin: 0;
  font-family: var(--font-serif);
  background: var(--bg);
  color: var(--fg);
  line-height: 1.7;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}

/* --- Skip link --- */
.skip-link {
  position: absolute;
  top: -100px;
  left: 8px;
  background: var(--accent);
  color: #fff;
  padding: 8px 12px;
  border-radius: 4px;
  z-index: 1000;
  text-decoration: none;
  font-family: var(--font-sans);
}
.skip-link:focus { top: 8px; }

/* --- Topbar --- */
.topbar {
  position: sticky;
  top: 0;
  height: calc(var(--topbar-h) + env(safe-area-inset-top));
  padding-top: env(safe-area-inset-top);
  background: color-mix(in srgb, var(--panel) 92%, transparent);
  backdrop-filter: saturate(1.4) blur(12px);
  -webkit-backdrop-filter: saturate(1.4) blur(12px);
  border-bottom: 1px solid var(--rule);
  display: flex;
  align-items: center;
  gap: 12px;
  padding-left: 16px;
  padding-right: 16px;
  z-index: 50;
  font-family: var(--font-sans);
}

.topbar .menu-btn {
  display: none;
  background: none;
  border: 0;
  color: var(--fg);
  font-size: 22px;
  cursor: pointer;
  padding: 8px;
  border-radius: 6px;
  min-width: 44px;
  min-height: 44px;
  touch-action: manipulation;
}
.topbar .menu-btn:hover { background: var(--bg-soft); }

.topbar .brand {
  font-weight: 600;
  letter-spacing: -0.01em;
  font-size: 0.98em;
  color: var(--accent);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 0 1 auto;
}

.topbar .progress-wrap {
  flex: 1 1 auto;
  height: 4px;
  background: var(--rule);
  border-radius: 2px;
  overflow: hidden;
  max-width: 320px;
  margin: 0 12px;
}
.topbar .progress-bar {
  height: 100%;
  background: linear-gradient(90deg, var(--accent), var(--violet));
  width: 0%;
  transition: width 120ms linear;
}

.topbar .search-wrap {
  position: relative;
  flex: 0 0 auto;
}
.topbar .search {
  background: var(--bg-soft);
  border: 1px solid var(--rule);
  color: var(--fg);
  font-family: var(--font-sans);
  font-size: 15px;
  padding: 8px 12px;
  border-radius: 8px;
  width: 220px;
  min-height: 40px;
}
.topbar .search:focus {
  outline: 2px solid var(--accent);
  outline-offset: 1px;
  background: var(--panel);
}

.topbar .filters {
  display: flex;
  gap: 6px;
}
.topbar .filter-btn {
  border: 1px solid var(--rule);
  background: var(--panel);
  color: var(--fg);
  padding: 6px 10px;
  border-radius: 999px;
  cursor: pointer;
  font-size: 14px;
  line-height: 1;
  min-height: 36px;
  min-width: 44px;
  font-family: var(--font-sans);
  touch-action: manipulation;
  transition: background var(--transition), color var(--transition);
}
.topbar .filter-btn[aria-pressed="true"] {
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
}
.topbar .filter-btn:hover { background: var(--bg-soft); }
.topbar .filter-btn[aria-pressed="true"]:hover { background: var(--accent-hover); }

.topbar .theme-btn,
.topbar .help-btn {
  border: 0;
  background: none;
  cursor: pointer;
  font-size: 18px;
  padding: 8px;
  border-radius: 6px;
  color: var(--fg);
  min-width: 44px;
  min-height: 44px;
  touch-action: manipulation;
}
.topbar .theme-btn:hover,
.topbar .help-btn:hover { background: var(--bg-soft); }

/* --- Layout shell --- */
.shell {
  display: grid;
  grid-template-columns: var(--sidebar-w) 1fr;
  gap: 0;
  max-width: 1600px;
  margin: 0 auto;
}

.sidebar {
  position: sticky;
  top: calc(var(--topbar-h) + env(safe-area-inset-top));
  height: calc(100vh - var(--topbar-h) - env(safe-area-inset-top));
  overflow-y: auto;
  border-right: 1px solid var(--rule);
  padding: 18px 14px 24px;
  background: var(--panel);
  font-family: var(--font-sans);
}

.sidebar h2 {
  font-size: 0.75em;
  text-transform: uppercase;
  letter-spacing: 0.14em;
  color: var(--fg-soft);
  margin: 0 0 10px;
  font-weight: 600;
}

.toc-root, .toc-root ol {
  list-style: none;
  padding: 0;
  margin: 0;
}
.toc-root ol {
  padding-left: 14px;
  border-left: 1px solid var(--rule);
  margin: 4px 0 4px 6px;
}
.toc-entry > a {
  display: flex;
  align-items: center;
  gap: 8px;
  text-decoration: none;
  color: var(--fg);
  padding: 6px 10px;
  border-radius: 6px;
  font-size: 0.92em;
  line-height: 1.4;
  min-height: 34px;
  transition: background var(--transition), color var(--transition);
}
.toc-entry > a:hover {
  background: var(--bg-soft);
  color: var(--accent);
}
.toc-entry > a[aria-current="location"] {
  background: var(--violet-soft);
  color: var(--accent);
  font-weight: 600;
}
.toc-entry .toc-check {
  opacity: 0;
  font-size: 0.85em;
  color: #3a9a3a;
  margin-left: auto;
  flex-shrink: 0;
}
.toc-entry.read > a .toc-check { opacity: 1; }
.toc-entry.toc-part > a {
  font-weight: 700;
  color: var(--accent);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 0.82em;
  margin-top: 14px;
}
.toc-entry.toc-depth-1 > a { font-weight: 600; }
.toc-entry.hidden { display: none; }
.toc-entry mark {
  background: var(--badge-partial-bg);
  color: var(--badge-partial-fg);
  padding: 0 2px;
  border-radius: 2px;
}

/* --- Main content --- */
.main {
  padding: 30px 40px 120px;
  min-width: 0;
  max-width: 880px;
  margin: 0 auto;
}

.chapter {
  max-width: 72ch;
  margin: 0 auto;
  padding-top: 12px;
  scroll-margin-top: calc(var(--topbar-h) + 24px);
}
.chapter + .chapter {
  margin-top: 48px;
  padding-top: 48px;
  border-top: 1px solid var(--rule);
}

.chapter section[id] { scroll-margin-top: calc(var(--topbar-h) + 16px); }

h1, h2, h3, h4, h5, h6 {
  font-family: var(--font-sans);
  color: var(--accent);
  letter-spacing: -0.01em;
  margin: 1.8em 0 0.6em;
  line-height: 1.25;
}
h1 {
  font-size: 1.9em;
  border-bottom: 2px solid var(--accent);
  padding-bottom: 0.2em;
  margin-top: 0.2em;
}
h2 {
  font-size: 1.45em;
  border-bottom: 1px solid var(--rule);
  padding-bottom: 0.15em;
}
h3 { font-size: 1.2em; }
h4 { font-size: 1.05em; }
h5, h6 { font-size: 1em; font-style: italic; color: var(--fg-soft); }

p { margin: 0.7em 0; }
strong { color: var(--fg); font-weight: 600; }
em { font-style: italic; }

a {
  color: var(--accent);
  text-decoration: none;
  border-bottom: 1px dotted var(--accent-soft);
  transition: border-bottom-style var(--transition), color var(--transition);
}
a:hover { border-bottom-style: solid; color: var(--accent-hover); }

/* Blockquote */
blockquote {
  margin: 1.4em 0;
  padding: 0.9em 1.2em;
  border-left: 4px solid var(--accent-soft);
  background: var(--bg-soft);
  font-style: italic;
  border-radius: 2px;
}
blockquote p { margin: 0.3em 0; }

/* Lists */
ul, ol { padding-left: 1.6em; margin: 0.6em 0; }
li { margin: 0.25em 0; }

/* Code */
code {
  font-family: var(--font-mono);
  font-size: 0.88em;
  background: var(--code-bg);
  color: var(--code-fg);
  padding: 0.1em 0.35em;
  border-radius: 3px;
  word-break: break-word;
}
pre {
  position: relative;
  font-family: var(--font-mono);
  font-size: 0.88em;
  background: var(--code-bg);
  border: 1px solid var(--rule);
  border-radius: 6px;
  padding: 14px 16px;
  margin: 1em 0;
  overflow-x: auto;
  line-height: 1.5;
}
pre code { background: none; padding: 0; font-size: 1em; color: var(--code-fg); }
pre[data-lang]::before {
  content: attr(data-lang);
  position: absolute;
  top: 6px;
  right: 56px;
  font-family: var(--font-sans);
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--fg-soft);
}
.copy-btn {
  position: absolute;
  top: 6px;
  right: 8px;
  background: var(--panel);
  border: 1px solid var(--rule);
  color: var(--fg);
  font-family: var(--font-sans);
  font-size: 11px;
  padding: 4px 8px;
  border-radius: 4px;
  cursor: pointer;
  opacity: 0.85;
  min-height: 28px;
  touch-action: manipulation;
  transition: opacity var(--transition), background var(--transition);
}
.copy-btn:hover { opacity: 1; background: var(--bg-soft); }

/* Tables */
.table-wrap {
  overflow-x: auto;
  margin: 1.2em 0;
  border-radius: 4px;
  border: 1px solid var(--rule);
}
table {
  border-collapse: collapse;
  width: 100%;
  font-family: var(--font-sans);
  font-size: 0.94em;
}
thead {
  background: var(--accent);
  color: #fff;
  position: sticky;
  top: 0;
}
th, td {
  padding: 0.55em 0.8em;
  border-bottom: 1px solid var(--rule);
  text-align: left;
  vertical-align: top;
}
tbody tr:nth-child(even) { background: var(--bg-soft); }

/* Figures */
figure {
  margin: 1.6em 0;
  text-align: center;
}
figure img {
  max-width: 100%;
  height: auto;
  border-radius: 4px;
  cursor: zoom-in;
  box-shadow: var(--shadow);
}
figcaption {
  font-family: var(--font-sans);
  font-size: 0.88em;
  color: var(--fg-soft);
  margin-top: 0.5em;
  font-style: italic;
}
p img, img {
  max-width: 100%;
  height: auto;
  cursor: zoom-in;
  border-radius: 4px;
}

/* Math */
math { font-family: "Latin Modern Math", "Cambria Math", var(--font-serif); }
math[display="block"] {
  display: block;
  overflow-x: auto;
  overflow-y: hidden;
  padding: 0.4em 0;
  margin: 0.9em 0;
  text-align: center;
}
.math-inline, math[display="inline"] { white-space: nowrap; }

/* Update-box */
details.update-box {
  background: var(--update-bg);
  border-left: 4px solid var(--update-border);
  border-radius: 4px;
  margin: 1.6em 0;
  padding: 8px 18px 8px;
  overflow: hidden;
}
details.update-box > summary {
  cursor: pointer;
  list-style: none;
  padding: 6px 0;
  font-family: var(--font-sans);
  user-select: none;
  display: flex;
  align-items: center;
  gap: 10px;
}
details.update-box > summary::-webkit-details-marker { display: none; }
details.update-box > summary::before {
  content: "📘";
  font-size: 1.1em;
  flex-shrink: 0;
}
details.update-box > summary > :is(h2, h3, h4) {
  margin: 0;
  border: 0;
  padding: 0;
  font-size: 1.15em;
  color: var(--update-border);
}
details.update-box[open] > summary { margin-bottom: 6px; }
details.update-box > *:not(summary) { margin-left: 6px; }

/* Status badges */
.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 8px;
  border-radius: 999px;
  font-family: var(--font-sans);
  font-size: 0.78em;
  font-weight: 600;
  margin: 0 2px;
  white-space: nowrap;
  vertical-align: baseline;
}
.status-badge.status-current { background: var(--badge-current-bg); color: var(--badge-current-fg); }
.status-badge.status-partial { background: var(--badge-partial-bg); color: var(--badge-partial-fg); }
.status-badge.status-legacy { background: var(--badge-legacy-bg); color: var(--badge-legacy-fg); }

/* Search highlight */
.chapter mark.search-hit {
  background: var(--badge-partial-bg);
  color: var(--badge-partial-fg);
  padding: 0 2px;
  border-radius: 2px;
}

/* Status filtering */
body[data-filter="current"] .chapter p:not(:has(.status-current)):not(:has(strong)):is(.legacy-only),
body[data-filter="current"] .filter-hide-current { display: none !important; }

/* Chapter nav */
.chapter-nav {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  margin: 48px auto 0;
  max-width: 72ch;
  font-family: var(--font-sans);
}
.chapter-nav a {
  display: flex;
  flex-direction: column;
  padding: 12px 16px;
  border: 1px solid var(--rule);
  border-radius: 8px;
  background: var(--panel);
  color: var(--fg);
  text-decoration: none;
  flex: 1 1 45%;
  max-width: 48%;
  min-height: 60px;
  border-bottom: 1px solid var(--rule);
  transition: border-color var(--transition), background var(--transition);
}
.chapter-nav a.next { text-align: right; }
.chapter-nav a:hover { border-color: var(--accent); background: var(--bg-soft); }
.chapter-nav .label {
  font-size: 0.75em;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--fg-soft);
}
.chapter-nav .ctitle {
  color: var(--accent);
  font-weight: 600;
  margin-top: 2px;
}

/* Lightbox */
.lightbox {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.88);
  display: none;
  align-items: center;
  justify-content: center;
  z-index: 200;
  padding: env(safe-area-inset-top) 16px env(safe-area-inset-bottom);
  cursor: zoom-out;
}
.lightbox.open { display: flex; }
.lightbox img {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  border-radius: 4px;
  box-shadow: 0 8px 40px rgba(0,0,0,0.5);
  cursor: default;
}
.lightbox .close {
  position: absolute;
  top: calc(env(safe-area-inset-top) + 12px);
  right: 16px;
  background: rgba(255,255,255,0.14);
  color: #fff;
  border: 0;
  width: 44px;
  height: 44px;
  border-radius: 50%;
  font-size: 22px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
}
.lightbox .close:hover { background: rgba(255,255,255,0.26); }

/* Back to top */
.back-top {
  position: fixed;
  right: 20px;
  bottom: calc(20px + env(safe-area-inset-bottom));
  background: var(--accent);
  color: #fff;
  border: 0;
  width: 48px;
  height: 48px;
  border-radius: 50%;
  font-size: 20px;
  cursor: pointer;
  box-shadow: var(--shadow);
  display: none;
  align-items: center;
  justify-content: center;
  z-index: 60;
}
.back-top.show { display: flex; }

/* Help overlay */
.help-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.55);
  display: none;
  align-items: center;
  justify-content: center;
  z-index: 210;
  padding: 16px;
}
.help-overlay.open { display: flex; }
.help-overlay .card {
  background: var(--panel);
  color: var(--fg);
  padding: 24px 28px;
  border-radius: 12px;
  max-width: 480px;
  width: 100%;
  font-family: var(--font-sans);
  box-shadow: var(--shadow);
}
.help-overlay h3 { margin: 0 0 12px; color: var(--accent); border: 0; }
.help-overlay dl { display: grid; grid-template-columns: auto 1fr; gap: 6px 14px; margin: 0; }
.help-overlay kbd {
  font-family: var(--font-mono);
  font-size: 0.85em;
  background: var(--bg-soft);
  border: 1px solid var(--rule);
  border-bottom-width: 2px;
  border-radius: 3px;
  padding: 1px 6px;
}

/* Drawer overlay */
.drawer-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.5);
  z-index: 40;
  opacity: 0;
  pointer-events: none;
  transition: opacity var(--transition);
}
body.drawer-open .drawer-backdrop { opacity: 1; pointer-events: auto; }

/* Responsive */
@media (max-width: 900px) {
  :root { --sidebar-w: 300px; }
  .shell { grid-template-columns: 1fr; }
  .sidebar {
    position: fixed;
    left: 0;
    top: 0;
    height: 100vh;
    width: var(--sidebar-w);
    z-index: 45;
    transform: translateX(-100%);
    transition: transform var(--transition);
    padding-top: calc(env(safe-area-inset-top) + 16px);
    padding-bottom: calc(env(safe-area-inset-bottom) + 16px);
  }
  body.drawer-open .sidebar { transform: translateX(0); }
  .topbar .menu-btn { display: inline-flex; align-items: center; justify-content: center; }
  .main { padding: 20px 18px 90px; }
  .topbar .brand { font-size: 0.9em; }
  .topbar .progress-wrap { max-width: none; }
  .topbar .filters { display: none; }
  .topbar .search { width: 120px; }
}

@media (max-width: 600px) {
  .topbar { gap: 6px; padding-left: 6px; padding-right: 6px; }
  .topbar .brand { max-width: 40vw; font-size: 0.85em; }
  .topbar .search { width: 100px; font-size: 14px; }
  .topbar .help-btn { display: none; }
  .main { padding: 16px 14px 80px; }
  .chapter { padding-top: 6px; }
  h1 { font-size: 1.5em; }
  h2 { font-size: 1.22em; }
  h3 { font-size: 1.08em; }
  .chapter-nav { flex-direction: column; }
  .chapter-nav a { max-width: 100%; }
  body { line-height: 1.6; }
}

/* Focus rings */
:focus-visible {
  outline: 2px solid var(--accent);
  outline-offset: 2px;
  border-radius: 4px;
}

/* Selection */
::selection {
  background: var(--violet-soft);
  color: var(--fg);
}

/* Print */
@media print {
  .topbar, .sidebar, .back-top, .help-overlay, .drawer-backdrop, .copy-btn { display: none !important; }
  .shell { grid-template-columns: 1fr; }
  .main { max-width: none; padding: 0; }
  a { color: var(--fg); border: 0; }
}
"""


JS_TEMPLATE = r"""
(function(){
  'use strict';
  const $ = (s, r=document) => r.querySelector(s);
  const $$ = (s, r=document) => Array.from(r.querySelectorAll(s));
  const LS_THEME = 'dl-opt-theme';
  const LS_READ = 'dl-opt-read';
  const LS_PROGRESS = 'dl-opt-progress';
  const mqlCoarse = matchMedia('(hover: none) and (pointer: coarse)');

  // --- Theme ---
  function setTheme(t){
    document.documentElement.setAttribute('data-theme', t);
    localStorage.setItem(LS_THEME, t);
  }
  function initTheme(){
    const saved = localStorage.getItem(LS_THEME);
    if (saved) { setTheme(saved); return; }
    const prefersDark = matchMedia('(prefers-color-scheme: dark)').matches;
    setTheme(prefersDark ? 'dark' : 'light');
  }
  initTheme();

  document.addEventListener('DOMContentLoaded', () => {
    const main = $('#main');
    const sidebar = $('#sidebar');
    const topbar = $('.topbar');
    const search = $('#search');
    const progressBar = $('#progress-bar');
    const chapters = $$('.chapter', main);
    const tocLinks = $$('.toc-entry > a', sidebar);

    // --- Menu toggle (drawer on mobile) ---
    const menuBtn = $('#menu-btn');
    const backdrop = $('.drawer-backdrop');
    function toggleDrawer(open){
      const v = open ?? !document.body.classList.contains('drawer-open');
      document.body.classList.toggle('drawer-open', v);
      menuBtn?.setAttribute('aria-expanded', String(v));
    }
    menuBtn?.addEventListener('click', () => toggleDrawer());
    backdrop?.addEventListener('click', () => toggleDrawer(false));

    // --- Theme toggle ---
    $('#theme-btn')?.addEventListener('click', () => {
      const cur = document.documentElement.getAttribute('data-theme');
      setTheme(cur === 'dark' ? 'light' : 'dark');
    });

    // --- Wrap tables for horizontal scroll ---
    $$('.chapter table').forEach(t => {
      if (t.parentElement.classList.contains('table-wrap')) return;
      const wrap = document.createElement('div');
      wrap.className = 'table-wrap';
      t.parentNode.insertBefore(wrap, t);
      wrap.appendChild(t);
    });

    // --- Copy button on <pre> ---
    $$('.chapter pre').forEach(pre => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'copy-btn';
      btn.textContent = 'copiar';
      btn.addEventListener('click', async () => {
        const code = pre.querySelector('code')?.innerText || pre.innerText;
        try {
          await navigator.clipboard.writeText(code);
          btn.textContent = '✓ copiado';
          setTimeout(() => { btn.textContent = 'copiar'; }, 1500);
        } catch(e) {
          btn.textContent = 'erro';
        }
      });
      pre.appendChild(btn);
    });

    // --- Build chapter nav (prev/next) ---
    const chapterIds = chapters.map(c => c.dataset.chapterId);
    const chapterTitles = chapters.map(c => {
      const h = c.querySelector('h1, h2, h3');
      return h ? h.textContent.trim() : c.dataset.chapterId;
    });
    chapters.forEach((c, i) => {
      const nav = document.createElement('nav');
      nav.className = 'chapter-nav';
      nav.setAttribute('aria-label', 'Navegação de capítulo');
      const prev = i > 0 ? { id: chapterIds[i-1], title: chapterTitles[i-1] } : null;
      const next = i < chapters.length - 1 ? { id: chapterIds[i+1], title: chapterTitles[i+1] } : null;
      if (prev) {
        const a = document.createElement('a');
        a.href = '#' + prev.id;
        a.className = 'prev';
        a.innerHTML = '<span class="label">← Anterior</span><span class="ctitle"></span>';
        a.querySelector('.ctitle').textContent = prev.title;
        nav.appendChild(a);
      } else {
        nav.appendChild(document.createElement('span'));
      }
      if (next) {
        const a = document.createElement('a');
        a.href = '#' + next.id;
        a.className = 'next';
        a.innerHTML = '<span class="label">Próximo →</span><span class="ctitle"></span>';
        a.querySelector('.ctitle').textContent = next.title;
        nav.appendChild(a);
      }
      c.appendChild(nav);
    });

    // --- TOC highlight via IntersectionObserver ---
    const idToLink = new Map();
    tocLinks.forEach(a => {
      const target = a.getAttribute('data-toc-target');
      if (target) idToLink.set(target, a);
    });
    const allTargets = [];
    chapters.forEach(c => {
      allTargets.push(c);
      c.querySelectorAll('section[id]').forEach(s => allTargets.push(s));
    });
    let currentActive = null;
    const io = new IntersectionObserver(entries => {
      const visible = entries.filter(e => e.isIntersecting)
        .sort((a,b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (!visible.length) return;
      const first = visible[0].target;
      const id = first.id || first.dataset.chapterId;
      const link = idToLink.get(id);
      if (link && link !== currentActive) {
        currentActive?.removeAttribute('aria-current');
        link.setAttribute('aria-current', 'location');
        currentActive = link;
      }
    }, { rootMargin: '-80px 0px -60% 0px', threshold: 0 });
    allTargets.forEach(t => io.observe(t));

    // --- Progress bar + back-to-top + deep linking ---
    const backTop = $('#back-top');
    function updateProgress(){
      const h = document.documentElement;
      const total = h.scrollHeight - h.clientHeight;
      const cur = h.scrollTop;
      const pct = total > 0 ? (cur / total) * 100 : 0;
      if (progressBar) progressBar.style.width = pct.toFixed(1) + '%';
      if (backTop) backTop.classList.toggle('show', cur > 600);
    }
    window.addEventListener('scroll', updateProgress, { passive: true });
    updateProgress();
    backTop?.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));

    // --- Chapter-read tracking ---
    const readSet = new Set(JSON.parse(localStorage.getItem(LS_READ) || '[]'));
    function markRead(id){
      if (readSet.has(id)) return;
      readSet.add(id);
      localStorage.setItem(LS_READ, JSON.stringify([...readSet]));
      idToLink.get(id)?.parentElement?.classList.add('read');
    }
    readSet.forEach(id => idToLink.get(id)?.parentElement?.classList.add('read'));
    const readIo = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (e.isIntersecting && e.intersectionRatio > 0.65) {
          markRead(e.target.dataset.chapterId);
        }
      });
    }, { threshold: [0, 0.3, 0.65, 1] });
    chapters.forEach(c => readIo.observe(c));

    // --- Search ---
    const searchIndex = chapters.map(c => {
      return {
        id: c.dataset.chapterId,
        text: c.innerText.toLowerCase(),
        title: (c.querySelector('h1,h2,h3')?.textContent || '').toLowerCase(),
      };
    });

    function clearHighlights(){
      $$('.chapter mark.search-hit').forEach(m => {
        const parent = m.parentNode;
        parent.replaceChild(document.createTextNode(m.textContent), m);
        parent.normalize();
      });
    }
    function highlight(term){
      if (!term || term.length < 2) return;
      const re = new RegExp('(' + term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'gi');
      const walker = document.createTreeWalker(main, NodeFilter.SHOW_TEXT, {
        acceptNode: n => {
          if (!n.nodeValue.trim()) return NodeFilter.FILTER_REJECT;
          const p = n.parentElement;
          if (!p || ['SCRIPT','STYLE','MARK','PRE','CODE'].includes(p.tagName)) return NodeFilter.FILTER_REJECT;
          if (p.closest('math')) return NodeFilter.FILTER_REJECT;
          return re.test(n.nodeValue) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
        }
      });
      const nodes = [];
      while (walker.nextNode()) nodes.push(walker.currentNode);
      nodes.forEach(n => {
        const frag = document.createDocumentFragment();
        const parts = n.nodeValue.split(re);
        parts.forEach(p => {
          if (!p) return;
          if (re.test(p)) {
            const m = document.createElement('mark');
            m.className = 'search-hit';
            m.textContent = p;
            frag.appendChild(m);
          } else {
            frag.appendChild(document.createTextNode(p));
          }
          re.lastIndex = 0;
        });
        n.parentNode.replaceChild(frag, n);
      });
    }

    function resetTocTitles(){
      tocLinks.forEach(a => {
        const tit = a.querySelector('.toc-title');
        if (tit) tit.textContent = tit.textContent; // collapse any <mark>
      });
    }

    let searchTimer;
    search?.addEventListener('input', () => {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(() => {
        const term = search.value.trim().toLowerCase();
        clearHighlights();
        resetTocTitles();
        if (!term) {
          tocLinks.forEach(a => a.parentElement.classList.remove('hidden'));
          return;
        }
        const escaped = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const reTest = new RegExp(escaped, 'i');
        const reMark = new RegExp(escaped, 'gi');
        tocLinks.forEach(a => {
          const tit = a.querySelector('.toc-title');
          const text = tit ? tit.textContent : a.textContent;
          const match = reTest.test(text);
          a.parentElement.classList.toggle('hidden', !match);
          if (tit && match) tit.innerHTML = text.replace(reMark, m => `<mark>${m}</mark>`);
        });
        highlight(term);
      }, 160);
    });

    // --- Status filters ---
    $$('.filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const status = btn.dataset.status;
        const pressed = btn.getAttribute('aria-pressed') === 'true';
        $$('.filter-btn').forEach(b => b.setAttribute('aria-pressed', 'false'));
        if (pressed) {
          document.body.removeAttribute('data-filter');
        } else {
          btn.setAttribute('aria-pressed', 'true');
          document.body.setAttribute('data-filter', status);
          applyStatusFilter(status);
          return;
        }
        applyStatusFilter(null);
      });
    });

    function applyStatusFilter(status){
      const all = $$('.chapter p, .chapter li, .chapter td, .chapter tr, .chapter h1, .chapter h2, .chapter h3, .chapter h4, .chapter h5');
      all.forEach(el => el.classList.remove('filter-dimmed'));
      if (!status) return;
      const wanted = 'status-' + status;
      all.forEach(el => {
        const badges = el.querySelectorAll('.status-badge');
        if (badges.length === 0) return; // unmarked content stays at full opacity
        const match = [...badges].some(b => b.classList.contains(wanted));
        if (!match) el.classList.add('filter-dimmed');
      });
    }
    // CSS for dimming filter (injected)
    const dimStyle = document.createElement('style');
    dimStyle.textContent = '.filter-dimmed{opacity:.28;transition:opacity 160ms;} .filter-dimmed:hover{opacity:1;}';
    document.head.appendChild(dimStyle);

    // --- Lightbox ---
    const lightbox = $('#lightbox');
    const lbImg = lightbox?.querySelector('img');
    function openLightbox(src, alt){
      if (!lightbox) return;
      lbImg.src = src;
      lbImg.alt = alt || '';
      lightbox.classList.add('open');
      lightbox.setAttribute('aria-hidden', 'false');
    }
    function closeLightbox(){
      lightbox?.classList.remove('open');
      lightbox?.setAttribute('aria-hidden', 'true');
      if (lbImg) lbImg.src = '';
    }
    $$('.chapter img').forEach(img => {
      img.addEventListener('click', e => {
        e.preventDefault();
        openLightbox(img.currentSrc || img.src, img.alt);
      });
    });
    lightbox?.addEventListener('click', e => {
      if (e.target === lightbox || e.target.classList.contains('close')) closeLightbox();
    });

    // --- Keyboard shortcuts (desktop only) ---
    if (!mqlCoarse.matches) {
      document.addEventListener('keydown', e => {
        const tag = e.target.tagName;
        const typing = tag === 'INPUT' || tag === 'TEXTAREA' || e.target.isContentEditable;
        if (e.key === 'Escape') {
          if (lightbox?.classList.contains('open')) closeLightbox();
          else if ($('.help-overlay')?.classList.contains('open')) closeHelp();
          else if (document.body.classList.contains('drawer-open')) toggleDrawer(false);
          else if (search && document.activeElement === search) { search.blur(); search.value=''; search.dispatchEvent(new Event('input')); }
          return;
        }
        if (typing) return;
        if (e.key === '/') { e.preventDefault(); search?.focus(); return; }
        if (e.key === '?') { openHelp(); return; }
        if (e.key === 'd') { $('#theme-btn')?.click(); return; }
        if (e.key === 't') { toggleDrawer(); return; }
        if (e.key === 'j') { window.scrollBy({ top: 80, behavior: 'smooth' }); return; }
        if (e.key === 'k') { window.scrollBy({ top: -80, behavior: 'smooth' }); return; }
        if (e.key === 'ArrowRight') { gotoChapter(+1); return; }
        if (e.key === 'ArrowLeft') { gotoChapter(-1); return; }
      });
    }

    function gotoChapter(delta){
      const cur = [...chapters].filter(c => c.getBoundingClientRect().top <= 120).slice(-1)[0] || chapters[0];
      const idx = chapters.indexOf(cur);
      const target = chapters[Math.max(0, Math.min(chapters.length-1, idx + delta))];
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // --- Help overlay ---
    const help = $('#help');
    function openHelp(){ help?.classList.add('open'); help?.setAttribute('aria-hidden','false'); }
    function closeHelp(){ help?.classList.remove('open'); help?.setAttribute('aria-hidden','true'); }
    $('#help-btn')?.addEventListener('click', openHelp);
    help?.addEventListener('click', e => { if (e.target === help || e.target.classList.contains('close-help')) closeHelp(); });

    // --- Swipe navigation (mobile) ---
    if (mqlCoarse.matches) {
      let sx=0, sy=0, active=false;
      main.addEventListener('pointerdown', e => {
        if (e.pointerType !== 'touch') return;
        // Don't hijack horizontal scroll inside wrappers
        const scrollable = e.target.closest('.table-wrap, pre, math[display="block"]');
        if (scrollable && scrollable.scrollWidth > scrollable.clientWidth) return;
        sx = e.clientX; sy = e.clientY; active = true;
      });
      main.addEventListener('pointerup', e => {
        if (!active) return;
        active = false;
        const dx = e.clientX - sx;
        const dy = e.clientY - sy;
        if (Math.abs(dx) > 80 && Math.abs(dx) > Math.abs(dy) * 1.8) {
          gotoChapter(dx < 0 ? +1 : -1);
        }
      });
      main.addEventListener('pointercancel', () => { active = false; });
    }

    // --- Close drawer on TOC click (mobile) ---
    tocLinks.forEach(a => a.addEventListener('click', () => {
      if (matchMedia('(max-width: 900px)').matches) toggleDrawer(false);
    }));

    // --- Scroll to hash on load (IO can race) ---
    if (location.hash) {
      const el = document.getElementById(location.hash.slice(1));
      if (el) setTimeout(() => el.scrollIntoView({ block: 'start' }), 80);
    }

    // --- Save last-read chapter ---
    window.addEventListener('beforeunload', () => {
      localStorage.setItem(LS_PROGRESS, String(window.scrollY));
    });
  });

  // --- Register service worker ---
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
      navigator.serviceWorker.register('sw.js').catch(() => {/* no-op */});
    });
  }
})();
"""


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Otimização em Deep Learning — Interativo</title>
<meta name="description" content="Versão interativa do ebook Otimização em Deep Learning (Módulo 2 - MsC AI, UC Boulder).">
<link rel="manifest" href="manifest.webmanifest">
<meta name="theme-color" content="#1a3a5c" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#0f1419" media="(prefers-color-scheme: dark)">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="DL Optimization">
<link rel="apple-touch-icon" href="media/apple-touch-icon.png">
<link rel="icon" type="image/png" sizes="192x192" href="media/icon-192.png">
<style>{css}</style>
</head>
<body>
<a href="#main" class="skip-link">Pular para o conteúdo</a>

<header class="topbar" role="banner">
  <button type="button" id="menu-btn" class="menu-btn" aria-label="Abrir sumário" aria-expanded="false" aria-controls="sidebar">☰</button>
  <a href="#ch001" class="brand">Otimização em DL</a>
  <div class="progress-wrap" aria-hidden="true"><div id="progress-bar" class="progress-bar"></div></div>
  <div class="search-wrap">
    <input type="search" id="search" class="search" placeholder="Buscar…" aria-label="Buscar no ebook" autocomplete="off">
  </div>
  <div class="filters" role="group" aria-label="Filtros de status">
    <button type="button" class="filter-btn" data-status="current" aria-pressed="false" title="Só atual">🟢</button>
    <button type="button" class="filter-btn" data-status="partial" aria-pressed="false" title="Só parcial">🟡</button>
    <button type="button" class="filter-btn" data-status="legacy" aria-pressed="false" title="Só legado">🔴</button>
  </div>
  <button type="button" id="theme-btn" class="theme-btn" aria-label="Alternar tema claro/escuro" title="Tema (d)">◐</button>
  <button type="button" id="help-btn" class="help-btn" aria-label="Mostrar atalhos" title="Atalhos (?)">?</button>
</header>

<div class="drawer-backdrop" aria-hidden="true"></div>

<div class="shell">
  <aside id="sidebar" class="sidebar" aria-label="Sumário">
    <h2>Sumário</h2>
    {toc}
  </aside>

  <main id="main" class="main" role="main" tabindex="-1">
    {chapters}
  </main>
</div>

<button type="button" id="back-top" class="back-top" aria-label="Voltar ao topo" title="Topo">↑</button>

<div id="lightbox" class="lightbox" role="dialog" aria-modal="true" aria-label="Imagem ampliada" aria-hidden="true">
  <button type="button" class="close" aria-label="Fechar">✕</button>
  <img alt="">
</div>

<div id="help" class="help-overlay" role="dialog" aria-modal="true" aria-label="Atalhos de teclado" aria-hidden="true">
  <div class="card">
    <h3>Atalhos de teclado</h3>
    <dl>
      <dt><kbd>/</kbd></dt><dd>Focar busca</dd>
      <dt><kbd>d</kbd></dt><dd>Alternar dark mode</dd>
      <dt><kbd>t</kbd></dt><dd>Abrir/fechar sumário</dd>
      <dt><kbd>j</kbd>/<kbd>k</kbd></dt><dd>Scroll fino</dd>
      <dt><kbd>←</kbd>/<kbd>→</kbd></dt><dd>Capítulo anterior / próximo</dd>
      <dt><kbd>Esc</kbd></dt><dd>Fechar overlay / busca</dd>
      <dt><kbd>?</kbd></dt><dd>Esta legenda</dd>
    </dl>
    <p style="margin:16px 0 0;font-size:.85em;color:var(--fg-soft)">
      No mobile: swipe horizontal (≥80&nbsp;px) para próximo/anterior capítulo. Tap em imagem abre lightbox.
    </p>
    <button type="button" class="close-help" style="margin-top:14px;padding:8px 14px;border-radius:6px;border:1px solid var(--rule);background:var(--bg-soft);color:var(--fg);cursor:pointer">Fechar</button>
  </div>
</div>

<script>{js}</script>
<script>/* precache versioned by build script; updated on each build */
/* cache version: __CACHE_VERSION__ */</script>
</body>
</html>
"""


if __name__ == "__main__":
    build()
