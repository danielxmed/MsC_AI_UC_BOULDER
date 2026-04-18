"""Post-process extracted chapter markdowns.

- Trim §5.4 tail from ch01 and §12.9 tail from ch05 into a separate "fragments"
  chapter so the main chapters end cleanly.
- Rewrite image paths to `assets/<basename>`.
- Normalize headings: strip bold markers that pymupdf4llm adds (## **3.7 ...**).
- Remove page footer cruft (chapter names + page numbers) that leaks between pages.
"""

from __future__ import annotations

import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
CHAP_DIR = OUT / "base_chapters"

# ------- image path rewrite -------
IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def rewrite_images(md: str) -> str:
    def repl(m):
        alt, path = m.group(1), m.group(2)
        base = path.rsplit("/", 1)[-1]
        return f"![{alt}](assets/{base})"
    return IMG_RE.sub(repl, md)


# ------- remove bold markers around headings (## **12.4 Foo**) -------
HEAD_RE = re.compile(r"^(#{1,6})\s*\*{1,2}([^*\n]+?)\*{1,2}\s*$", re.MULTILINE)


def clean_headings(md: str) -> str:
    return HEAD_RE.sub(lambda m: f"{m.group(1)} {m.group(2).strip()}", md)


# ------- fence lines that look like python code comments misread as headings -------
CODE_MARKER_RE = re.compile(
    r"^(#[@ ].*(?:torch\.|d2l\.|nn\.|\.start\(|\.stop\(|timer\.|@save|"
    r"for \w+ in |def \w+|import |np\.|plt\.|matplotlib|Compute A = |hyperparams).*)$",
    re.MULTILINE,
)


def fence_code_comments(md: str) -> str:
    """Wrap lines like '# Compute A = BC ...' in backticks so they aren't parsed as H1."""
    def repl(m):
        line = m.group(1)
        return f"```python\n{line}\n```"
    return CODE_MARKER_RE.sub(repl, md)


# ------- scrub common D2L page footers / chapter captions -------
FOOTER_PATTERNS = [
    r"^Linear Neural Networks for Regression\s*$",
    r"^Builders[’']? Guide\s*$",
    r"^Optimization Algorithms\s*$",
    r"^Discussions\[\d+\]\s*\.?\s*$",
    r"^\d{1,3}\s*$",  # lone page numbers
]
FOOTER_RE = re.compile("|".join(FOOTER_PATTERNS), re.MULTILINE)


def strip_footers(md: str) -> str:
    md = FOOTER_RE.sub("", md)
    # Collapse >3 blank lines to 2
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md


# ------- split chapter at a heading boundary, returning (keep, fragment) -------
def split_before_heading(md: str, heading_substr: str) -> tuple[str, str]:
    """Find first heading line containing heading_substr; return (before, from_here)."""
    idx_match = re.search(
        rf"^#{{1,6}}\s.*{re.escape(heading_substr)}.*$",
        md,
        re.MULTILINE,
    )
    if idx_match is None:
        return md, ""
    i = idx_match.start()
    return md[:i].rstrip() + "\n", md[i:]


def main() -> None:
    fragments: list[tuple[str, str]] = []  # (title, content)

    for fp in sorted(CHAP_DIR.glob("*.md")):
        txt = fp.read_text()
        txt = clean_headings(txt)
        txt = rewrite_images(txt)
        txt = fence_code_comments(txt)
        txt = strip_footers(txt)

        if fp.name == "ch01_weight_decay.md":
            keep, frag = split_before_heading(txt, "5.4.3 Summary")
            txt = keep
            if frag:
                fragments.append(("Trecho residual de §5.4 (final do capítulo anterior no PDF)", frag))

        if fp.name == "ch05_momentum.md":
            keep, frag = split_before_heading(txt, "12.9.3 Summary")
            txt = keep
            if frag:
                fragments.append(("Trecho residual de §12.9 (final do capítulo anterior no PDF)", frag))

        fp.write_text(txt)
        print(f"cleaned {fp.name}: {len(txt.split())} words")

    if fragments:
        out = CHAP_DIR / "ch08_fragments.md"
        body = [
            "# Trechos Complementares do PDF-base",
            "",
            "> O PDF de leitura (`modulo_2_leitura_base_concatenada.pdf`) é uma "
            "concatenação de excertos do *Dive into Deep Learning*. Em alguns "
            "limites de capítulo ficaram pequenos pedaços de seções vizinhas "
            "que não constam do escopo principal do módulo. Por respeito à "
            "integralidade do material, estes fragmentos são preservados aqui.",
            "",
        ]
        for title, content in fragments:
            body.append(f"## {title}")
            body.append("")
            body.append(content.strip())
            body.append("")
        out.write_text("\n".join(body))
        print(f"wrote {out.name} with {len(fragments)} fragments")


if __name__ == "__main__":
    main()
