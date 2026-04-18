# build_ebook — Pipeline de construção do EPUB

Este diretório contém o toolchain para construir o
`deep_learning_modulo_2_ebook.epub` (na raiz do repositório) a partir do material
em `deep_learning_modulo_2_leitura_base/`.

## Estrutura

```
build_ebook/
├── fetch_d2l_chapters.py    # fetcha HTML de d2l.ai e converte para markdown limpo
├── split_updates.py         # fatia o markdown de atualizações por seção
├── assemble.py              # concatena tudo num ebook.md unificado
├── make_cover.py            # gera cover.png via Pillow
├── build.sh                 # orquestra tudo + pandoc → EPUB3 final
├── post_pandoc_epub.py      # pós-processa: renomeia .svgz → .svg
├── verify.py                # checagens de integridade pós-build
├── cover.png                # (gerada)
├── style.css                # CSS de leitura
├── metadata.yaml            # metadados EPUB (título, autor, idioma)
├── papers/                  # deep-dives dos 10 papers (committed)
├── d2l_cache/               # HTML-fonte cacheado (committed p/ reruns offline)
├── extract_pdf.py           # [DEPRECATED] pipeline antigo via PDF
├── postprocess_chapters.py  # [DEPRECATED] pareado com o acima
├── base_chapters/           # (.gitignored — gerado por fetch_d2l_chapters.py)
├── d2l_assets/              # (.gitignored — imagens SVG baixadas de d2l.ai)
├── assets/                  # (.gitignored — mescla de imagens)
├── updates/                 # (.gitignored — gerado por split_updates.py)
└── ebook.md                 # (.gitignored — markdown unificado intermediário)
```

## Como rebuildar do zero

```bash
# Dependências
apt-get install -y pandoc poppler-utils
pip install Pillow beautifulsoup4 lxml requests

# Pipeline (tudo orquestrado pelo build.sh)
bash build_ebook/build.sh
python3 build_ebook/verify.py
```

Os passos individuais que `build.sh` executa são:

```bash
python3 build_ebook/fetch_d2l_chapters.py   # HTML d2l.ai → base_chapters/*.md
python3 build_ebook/split_updates.py        # updates markdown → updates/*.md
python3 build_ebook/make_cover.py           # cover.png (se ainda não existe)
python3 build_ebook/assemble.py             # concat → ebook.md
pandoc ... ebook.md -o deep_learning_modulo_2_ebook.epub
python3 build_ebook/post_pandoc_epub.py     # .svgz → .svg p/ compat Apple Books
```

O resultado é `deep_learning_modulo_2_ebook.epub` na raiz do repositório.

## Design

- **Texto-base**: `fetch_d2l_chapters.py` baixa o HTML oficial de d2l.ai
  (CC BY-SA 4.0) para as seções §3.7, §5.5, §5.6, §12.4, §12.5, §12.6, §12.10
  e §12.11, preserva o canal PyTorch dos tabs de código, converte a matemática
  MathJax (`\(...\)` / `\[...\]`) para LaTeX nativo (`$...$` / `$$...$$`), e
  delega a pandoc a serialização para markdown. HTML fica cacheado em
  `d2l_cache/` para builds offline reprodutíveis.
- **Atualizações**: fatiadas do markdown-fonte por expressão regular conforme a
  numeração de seção nele.
- **Papers**: resumos substanciais (~700w cada) escritos em português por mim
  a partir de fetches do arXiv HTML via WebFetch.
- **EPUB**: pandoc com `epub3`, MathML para equações, CSS custom, capa embutida.
  Pós-processamento renomeia `.svgz` → `.svg` (pandoc sempre rebatiza assets SVG,
  o que quebra a renderização em alguns leitores).
- **Validação**: 11 checagens automáticas em `verify.py`, incluindo regressões
  específicas (matemática corrompida, `__init__` como bold, blocos `<pre>` sem
  `<code>`, headers de página vazando como parágrafos).

## Nota histórica

A primeira versão do pipeline extraía os capítulos-base via `pymupdf4llm`
contra o PDF (`extract_pdf.py` + `postprocess_chapters.py`, hoje marcados como
DEPRECATED). Isso gerou EPUBs mal-formatados: equações como `∥w∥[2]`, código
Python como prosa, `__init__` renderizado como `**init**`. A migração para
HTML-de-d2l.ai eliminou todos esses artefatos — math e código agora chegam
intactos ao EPUB final.

## Garantias de integridade

O PDF-base tem 16057 palavras (via `pdftotext`). O EPUB final tem 31679 palavras
— mais que o dobro, porque inclui:

- Todo o conteúdo do PDF-base (inglês).
- Todas as atualizações 2023→2026 (português).
- 10 deep-dives de papers (português).
- Apêndices, glossário bilíngue, mapa de decisão, referências.

Nenhum conteúdo do PDF-base é resumido ou omitido — todo o texto extraído é
concatenado diretamente no EPUB.
