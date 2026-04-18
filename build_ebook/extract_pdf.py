"""[DEPRECATED] PDF-based extraction of D2L excerpts.

The build pipeline no longer uses this script. `pymupdf4llm` against the
dense-math, dense-code PDF produced unreadable output (corrupt equations,
`__init__` rendered as `**init**`, code as prose). It was replaced by
`fetch_d2l_chapters.py`, which pulls the canonical HTML from d2l.ai and
converts it with pandoc.

Kept on disk for historical reference. Do not run from `build.sh`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import pymupdf
import pymupdf4llm

ROOT = Path(__file__).resolve().parent.parent
PDF = ROOT / "deep_learning_modulo_2_leitura_base" / "modulo_2_leitura_base_concatenada.pdf"
OUT_DIR = Path(__file__).resolve().parent
IMG_DIR = OUT_DIR / "extracted_images"
CHAP_DIR = OUT_DIR / "base_chapters"

IMG_DIR.mkdir(exist_ok=True)
CHAP_DIR.mkdir(exist_ok=True)

SECTIONS = [
    ("ch01_weight_decay",        "3.7",   "Weight Decay"),
    ("ch02_generalization",      "5.5",   "Generalization in Deep Learning"),
    ("ch03_dropout",             "5.6",   "Dropout"),
    ("ch04_sgd_minibatch",       "12.4",  "Stochastic Gradient Descent & Minibatch SGD"),
    ("ch05_momentum",            "12.6",  "Momentum"),
    ("ch06_adam",                "12.10", "Adam"),
    ("ch07_lr_scheduling",       "12.11", "Learning Rate Scheduling"),
]


def extract_images(doc: pymupdf.Document) -> dict[int, list[str]]:
    """Extract every embedded image, save as PNG/JPG, return {page_idx: [filenames]}."""
    per_page: dict[int, list[str]] = {}
    seen_xrefs: set[int] = set()
    for page_idx in range(doc.page_count):
        page = doc[page_idx]
        imgs = []
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            try:
                pix = pymupdf.Pixmap(doc, xref)
                if pix.n - pix.alpha >= 4:  # CMYK
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                ext = "png"
                fname = f"pdf_p{page_idx+1:03d}_x{xref}.{ext}"
                out = IMG_DIR / fname
                pix.save(out)
                pix = None
                imgs.append(fname)
            except Exception as e:
                print(f"  ! image extract failed p{page_idx} x{xref}: {e}")
        if imgs:
            per_page[page_idx] = imgs
    return per_page


SECTION_RE = re.compile(
    r"^(#{1,6})\s+(\d+\.\d+(?:\.\d+)?)\.?\s+(.*)$",
    re.MULTILINE,
)


def find_section_starts(page_markdowns: list[str]) -> dict[str, tuple[int, int]]:
    """For each target section code (e.g. '3.7'), find (page_idx, char_offset) of its heading.

    Returns a dict {section_code: (page_idx, line_idx_within_page)}.
    """
    hits: dict[str, list[tuple[int, int]]] = {s[1]: [] for s in SECTIONS}
    for pi, md in enumerate(page_markdowns):
        for m in SECTION_RE.finditer(md):
            code = m.group(2)
            if code in hits:
                line_idx = md[: m.start()].count("\n")
                hits[code].append((pi, line_idx))
    # take first occurrence per code
    starts: dict[str, tuple[int, int]] = {}
    for code, lst in hits.items():
        if lst:
            starts[code] = lst[0]
        else:
            print(f"  ! section {code} heading NOT found — will try loose match")
    return starts


def loose_find(page_markdowns: list[str], code: str, title: str) -> tuple[int, int] | None:
    """Fallback: find the section start by searching 'code' followed by title words."""
    pat = re.compile(rf"\b{re.escape(code)}\b[^\n]*{re.escape(title.split()[0])}", re.IGNORECASE)
    for pi, md in enumerate(page_markdowns):
        m = pat.search(md)
        if m:
            line_idx = md[: m.start()].count("\n")
            return pi, line_idx
    return None


def slice_chapter(page_markdowns: list[str], start: tuple[int, int], end: tuple[int, int] | None) -> str:
    """Return markdown slice from start (inclusive) to end (exclusive)."""
    sp, sl = start
    if end is None:
        ep, el = len(page_markdowns) - 1, None
    else:
        ep, el = end

    parts: list[str] = []
    for pi in range(sp, ep + 1):
        md = page_markdowns[pi]
        lines = md.splitlines()
        lo = sl if pi == sp else 0
        hi = (el if pi == ep and el is not None else len(lines))
        parts.append("\n".join(lines[lo:hi]))
    return "\n\n".join(parts).strip() + "\n"


def rewrite_image_refs(md: str, asset_map: dict[str, str]) -> str:
    """Rewrite ![...](image.png) to point into assets/ using asset_map.

    pymupdf4llm writes images with names it controls; map them into our extracted names.
    """
    def repl(m: re.Match) -> str:
        alt, path = m.group(1), m.group(2)
        basename = os.path.basename(path)
        new = asset_map.get(basename, basename)
        return f"![{alt}](assets/{new})"

    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl, md)


def main() -> None:
    print(f"Opening {PDF} ...")
    doc = pymupdf.open(PDF)
    print(f"  pages: {doc.page_count}")

    print("Extracting images from PDF ...")
    per_page_imgs = extract_images(doc)
    total_imgs = sum(len(v) for v in per_page_imgs.values())
    print(f"  saved {total_imgs} images into {IMG_DIR}")

    print("Running pymupdf4llm to_markdown (page_chunks=True) ...")
    chunks = pymupdf4llm.to_markdown(
        str(PDF),
        page_chunks=True,
        write_images=True,
        image_path=str(IMG_DIR),
        image_format="png",
        dpi=200,
        extract_words=False,
    )
    page_markdowns: list[str] = [c["text"] if isinstance(c, dict) else c for c in chunks]
    print(f"  got {len(page_markdowns)} page markdowns")

    # Save raw for debugging
    (OUT_DIR / "raw_pages.json").write_text(json.dumps(page_markdowns, indent=2, ensure_ascii=False))

    # Locate section starts
    starts = find_section_starts(page_markdowns)
    for slug, code, title in SECTIONS:
        if code not in starts:
            guess = loose_find(page_markdowns, code, title)
            if guess:
                starts[code] = guess
                print(f"  loose match {code} -> page {guess[0]+1}")
            else:
                print(f"  !! {code} still missing — check raw_pages.json")

    print("Starts detected:", {k: (v[0]+1, v[1]) for k, v in starts.items()})

    # Compute ordered section list with end offsets
    ordered = [(slug, code, title) for slug, code, title in SECTIONS if code in starts]
    ordered_sorted = sorted(ordered, key=lambda x: starts[x[1]])

    # Extract chapters
    chapter_paths: list[str] = []
    for i, (slug, code, title) in enumerate(ordered_sorted):
        s = starts[code]
        e = starts[ordered_sorted[i+1][1]] if i + 1 < len(ordered_sorted) else None
        body = slice_chapter(page_markdowns, s, e)
        header = f"# §{code} — {title}\n\n"
        out = CHAP_DIR / f"{slug}.md"
        out.write_text(header + body)
        chapter_paths.append(str(out))
        print(f"  wrote {out.name}: {len(body.split())} words")

    # Copy images to assets directory with clean names
    ASSETS = OUT_DIR / "assets"
    ASSETS.mkdir(exist_ok=True)
    for img in IMG_DIR.glob("*"):
        dst = ASSETS / img.name
        if not dst.exists():
            shutil.copy2(img, dst)
    print(f"  assets: {len(list(ASSETS.glob('*')))} files")

    # Copy update-images too
    UPD_IMGS = ROOT / "deep_learning_modulo_2_leitura_base" / "imagens_atualizacoes"
    for img in UPD_IMGS.iterdir():
        shutil.copy2(img, ASSETS / img.name)
    print(f"  assets (incl updates): {len(list(ASSETS.glob('*')))} files")


if __name__ == "__main__":
    main()
