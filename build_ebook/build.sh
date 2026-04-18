#!/usr/bin/env bash
# Build the unified EPUB from ebook.md via pandoc.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="deep_learning_modulo_2_ebook.epub"

# 1. Fetch (or reuse cached) D2L HTML, clean, emit base_chapters/*.md + assets/
python3 build_ebook/fetch_d2l_chapters.py

# 2. Slice the Portuguese updates markdown into per-chapter fragments
python3 build_ebook/split_updates.py

# 3. Ensure cover image exists
python3 build_ebook/make_cover.py

# 4. Assemble unified ebook.md
python3 build_ebook/assemble.py

# 5. pandoc → EPUB3
pandoc build_ebook/ebook.md \
  -o "$OUT" \
  --from=markdown+tex_math_dollars+tex_math_single_backslash+fenced_divs+footnotes+pipe_tables+auto_identifiers \
  --to=epub3 \
  --toc --toc-depth=2 \
  --mathml \
  --css=build_ebook/style.css \
  --epub-cover-image=build_ebook/cover.png \
  --resource-path=.:build_ebook:build_ebook/assets:build_ebook/d2l_assets:deep_learning_modulo_2_leitura_base \
  --split-level=1 \
  --metadata title="Otimização em Deep Learning" \
  --metadata subtitle="Módulo 2 — Base (Dive into Deep Learning) + Atualizações 2023→2026" \
  --metadata author="Compilado para o MsC AI · UC Boulder" \
  --metadata date="Abril de 2026" \
  --metadata lang="pt-BR"

# 6. Post-process EPUB: rename pandoc's .svgz back to .svg.
#    Pandoc's epub writer unconditionally renames every .svg asset to .svgz
#    (a legitimate extension for gzipped SVG, but these files are plain XML,
#    not gzipped — and some e-readers, notably Apple Books, won't recognize
#    .svgz without a Content-Encoding header).
python3 build_ebook/post_pandoc_epub.py "$OUT"

echo ""
ls -lh "$OUT"
echo ""
echo "EPUB built: $OUT"
