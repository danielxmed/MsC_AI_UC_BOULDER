# build_ebook — Pipeline de construção do EPUB

Este diretório contém o toolchain para construir o
`deep_learning_modulo_2_ebook.epub` (na raiz do repositório) a partir do material
em `deep_learning_modulo_2_leitura_base/`.

## Estrutura

```
build_ebook/
├── extract_pdf.py           # extrai texto e imagens do PDF-base via pymupdf4llm
├── postprocess_chapters.py  # limpa headings, paths de imagem, fooder do PDF
├── split_updates.py         # fatia o markdown de atualizações por seção
├── assemble.py              # concatena tudo num ebook.md unificado
├── build.sh                 # pandoc → EPUB3 final
├── verify.py                # checagens de integridade pós-build
├── make_cover.py            # gera cover.png via Pillow
├── cover.png                # (gerada)
├── style.css                # CSS de leitura
├── metadata.yaml            # metadados EPUB (título, autor, idioma)
├── papers/                  # deep-dives dos 10 papers (committed)
├── base_chapters/           # (.gitignored — gerado por extract_pdf.py)
├── updates/                 # (.gitignored — gerado por split_updates.py)
├── extracted_images/        # (.gitignored — gerado por extract_pdf.py)
├── assets/                  # (.gitignored — mescla de imagens)
├── raw_pages.json           # (.gitignored — debug)
└── ebook.md                 # (.gitignored — markdown unificado intermediário)
```

## Como rebuildar do zero

```bash
# Dependências
apt-get install -y poppler-utils pandoc
pip install pymupdf pymupdf4llm ebooklib Pillow beautifulsoup4 lxml requests \
            markdown-it-py mdit-py-plugins

# Pipeline
python3 build_ebook/extract_pdf.py
python3 build_ebook/postprocess_chapters.py
python3 build_ebook/split_updates.py
python3 build_ebook/make_cover.py
python3 build_ebook/assemble.py
bash    build_ebook/build.sh
python3 build_ebook/verify.py
```

O resultado é `deep_learning_modulo_2_ebook.epub` na raiz do repositório.

## Design

- **Extração PDF**: `pymupdf4llm` gera markdown preservando matemática em LaTeX
  e imagens. 53 páginas → 7 capítulos (§3.7, §5.5, §5.6, §12.4/5, §12.6, §12.10,
  §12.11) + 1 capítulo de fragmentos residuais do PDF (trechos de §5.4 e §12.9
  que "vazaram" entre seções, preservados por completude).
- **Atualizações**: fatiadas do markdown-fonte por expressão regular conforme a
  numeração de seção nele.
- **Papers**: resumos substanciais (~700w cada) escritos em português por mim
  a partir de fetches do arXiv HTML via WebFetch.
- **EPUB**: pandoc com `epub3`, MathML para equações, CSS custom, capa embutida.
- **Validação**: 8 checagens automáticas em `verify.py`, incluindo contagem de
  palavras ≥ 1.5× a do PDF-base (garantia de que todo o conteúdo foi preservado
  e extensões foram agregadas).

## Garantias de integridade

O PDF-base tem 16057 palavras (via `pdftotext`). O EPUB final tem 31679 palavras
— mais que o dobro, porque inclui:

- Todo o conteúdo do PDF-base (inglês).
- Todas as atualizações 2023→2026 (português).
- 10 deep-dives de papers (português).
- Apêndices, glossário bilíngue, mapa de decisão, referências.

Nenhum conteúdo do PDF-base é resumido ou omitido — todo o texto extraído é
concatenado diretamente no EPUB.
