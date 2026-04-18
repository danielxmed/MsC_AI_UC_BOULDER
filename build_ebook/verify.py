"""End-to-end verification of the built EPUB.

Checks layout integrity, that the 7 extension images are embedded, that the
Part I chapters rebuilt from d2l.ai have properly-fenced code and MathML
math, and that none of the known extraction regressions reappeared.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EPUB = ROOT / "deep_learning_modulo_2_ebook.epub"
PDF = ROOT / "deep_learning_modulo_2_leitura_base" / "modulo_2_leitura_base_concatenada.pdf"
UPD_IMGS = ROOT / "deep_learning_modulo_2_leitura_base" / "imagens_atualizacoes"

# Regressions we explicitly guard against.
BAD_PATTERNS = [
    re.compile(r"\*\*init\*\*"),            # __init__ rendered as bold
    re.compile(r"\*\*call\*\*"),
    re.compile(r"\*\*len\*\*"),
    re.compile(r"\[\(i\)\]"),               # y[(][i][)] artefact
    re.compile(r"∥w∥\[2\]"),                # corrupt norm rendering
    re.compile(r"ℓ2regularized"),          # broken spacing after subscript
    re.compile(r"ℓ2-regularized"),          # same, hyphenated form
]

HEADER_LEAK_PATTERNS = [
    # A bare <p> whose text is exactly a known page-header phrase.
    re.compile(
        r"<p>\s*(Weight Decay|Dropout|Generalization(?: in Deep Learning)?|"
        r"Momentum|Adam|Stochastic Gradient Descent|"
        r"Minibatch Stochastic Gradient Descent|Learning Rate Scheduling)"
        r"\s*</p>"
    ),
    # Reference-marker leaks like "<p>... [84]</p>" or "<p>[169]</p>"
    re.compile(r"<p>\s*\[\d{2,4}\]\s*</p>"),
]


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "\u2713" if ok else "\u2717"
    print(f"  {mark} {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    failures = 0
    print(f"\nVerifying {EPUB.name} ({EPUB.stat().st_size/1024:.0f} KB)\n")

    with zipfile.ZipFile(EPUB) as z:
        names = z.namelist()
        xhtml = [n for n in names if n.endswith(".xhtml")]
        imgs_in_epub = [n for n in names if n.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".svg"))]

        # 1. EPUB layout
        ok = ("mimetype" in names
              and "META-INF/container.xml" in names
              and any(n.endswith(".opf") for n in names))
        if not check("EPUB layout (mimetype, container, OPF)", ok):
            failures += 1

        # 2. Chapter count plausible
        if not check("Chapter count plausible", 15 <= len(xhtml) <= 40,
                     f"{len(xhtml)} xhtml files"):
            failures += 1

        # 3. Extension images (by content hash) all present
        source_hashes = {
            hashlib.md5(p.read_bytes()).hexdigest(): p.name
            for p in UPD_IMGS.iterdir()
        }
        epub_hashes = {hashlib.md5(z.read(n)).hexdigest() for n in imgs_in_epub}
        missing = {n for h, n in source_hashes.items() if h not in epub_hashes}
        if not check("All 7 extension images embedded (by content hash)",
                     not missing,
                     f"missing: {sorted(missing)}" if missing
                     else f"{len(source_hashes)} present"):
            failures += 1

        # 4. Word count sanity: ebook content ≥ 1.5× the PDF baseline
        pdf_words = int(subprocess.check_output(
            ["pdftotext", str(PDF), "-"], text=True).split().__len__())
        total_words = 0
        xhtml_contents: dict[str, str] = {}
        for n in xhtml:
            html = z.read(n).decode("utf-8", errors="ignore")
            xhtml_contents[n] = html
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            total_words += len(text.split())
        ratio = total_words / pdf_words
        if not check("Content volume >> PDF (all base preserved + extras)",
                     ratio >= 1.5,
                     f"{total_words} words vs {pdf_words} in PDF (ratio {ratio:.2f}x)"):
            failures += 1

        # 5. Custom CSS embedded
        if not check("Custom CSS embedded",
                     any(n.endswith(".css") for n in names)):
            failures += 1

        # 6. Cover present
        if not check("Cover image present",
                     any("cover" in n.lower() and n.endswith(".png") for n in names)):
            failures += 1

        # 7. No broken image references
        img_basenames = {Path(n).name for n in imgs_in_epub}
        broken = []
        for n, html in xhtml_contents.items():
            for m in re.finditer(r'<img[^>]+src="([^"]+)"', html):
                src_base = Path(m.group(1)).name
                if src_base not in img_basenames and "cover" not in src_base.lower():
                    broken.append((n, m.group(1)))
        if not check("No broken image references", not broken,
                     f"{len(broken)} broken" if broken else "all refs resolve"):
            for b in broken[:5]:
                print(f"     - in {b[0]}: {b[1]}")
            failures += 1

        # 8. No pre-regression artefacts anywhere
        offenders: list[tuple[str, str]] = []
        for n, html in xhtml_contents.items():
            for pat in BAD_PATTERNS:
                m = pat.search(html)
                if m:
                    offenders.append((n, pat.pattern))
                    break
        if not check("No known extraction regressions (math/code/init)",
                     not offenders,
                     f"{len(offenders)} files with artefacts"):
            for n, pat in offenders[:5]:
                print(f"     - {n}: {pat}")
            failures += 1

        # 9. No page-header-only paragraphs, no bracket-ref leaks
        header_offenders: list[tuple[str, str]] = []
        for n, html in xhtml_contents.items():
            for pat in HEADER_LEAK_PATTERNS:
                m = pat.search(html)
                if m:
                    header_offenders.append((n, m.group()[:80]))
                    break
        if not check("No page-header/reference-marker leaks",
                     not header_offenders,
                     f"{len(header_offenders)} leaks" if header_offenders else "clean"):
            for n, s in header_offenders[:5]:
                print(f"     - {n}: {s}")
            failures += 1

        # 10. MathML present (spot-check ch004/ch005/ch006 contain math)
        math_required = ["ch004.xhtml", "ch005.xhtml", "ch006.xhtml", "ch007.xhtml"]
        math_missing: list[str] = []
        for want in math_required:
            match = next((n for n in xhtml if n.endswith(want)), None)
            if match is None:
                continue
            cnt = xhtml_contents[match].count("<math ")
            if cnt < 5:
                math_missing.append(f"{want}({cnt} tags)")
        if not check("MathML tags present in base-chapter files",
                     not math_missing,
                     ", ".join(math_missing) if math_missing
                     else f"checked {len(math_required)} files"):
            failures += 1

        # 11. Every <pre> contains a <code> (no code-as-prose regression)
        pre_without_code = []
        for n, html in xhtml_contents.items():
            for m in re.finditer(r"<pre[^>]*>(.*?)</pre>", html, re.DOTALL):
                if "<code" not in m.group(1):
                    pre_without_code.append(n)
                    break
        if not check("Every <pre> block wraps a <code> element",
                     not pre_without_code,
                     f"{len(pre_without_code)} files with raw <pre>" if pre_without_code else "ok"):
            failures += 1

    print()
    if failures == 0:
        print("ALL CHECKS PASSED.")
        return 0
    print(f"{failures} CHECKS FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
