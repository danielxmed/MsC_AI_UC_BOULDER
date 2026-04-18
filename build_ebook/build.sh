#!/usr/bin/env bash
# Build the unified EPUB from ebook.md via pandoc.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="deep_learning_modulo_2_ebook.epub"

pandoc build_ebook/ebook.md \
  -o "$OUT" \
  --from=markdown+tex_math_dollars+tex_math_single_backslash+fenced_divs+footnotes+pipe_tables+auto_identifiers \
  --to=epub3 \
  --toc --toc-depth=2 \
  --mathml \
  --css=build_ebook/style.css \
  --epub-cover-image=build_ebook/cover.png \
  --resource-path=.:build_ebook:build_ebook/assets:deep_learning_modulo_2_leitura_base \
  --split-level=1 \
  --metadata title="Otimização em Deep Learning" \
  --metadata subtitle="Módulo 2 — Base (Dive into Deep Learning) + Atualizações 2023→2026" \
  --metadata author="Compilado para o MsC AI · UC Boulder" \
  --metadata date="Abril de 2026" \
  --metadata lang="pt-BR"

echo ""
ls -lh "$OUT"
echo ""
echo "EPUB built: $OUT"
