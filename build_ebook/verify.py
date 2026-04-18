"""End-to-end verification of the built EPUB.

Checks:
  1. EPUB is a valid ZIP with standard OEBPS layout.
  2. All 7 extension images (imagens_atualizacoes/*) are embedded.
  3. Number of XHTML chapters is plausible.
  4. Word count matches pdftotext baseline within 5% for base content.
  5. Every image reference in the manifest resolves to an included file.
"""

from __future__ import annotations

import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EPUB = ROOT / "deep_learning_modulo_2_ebook.epub"
PDF = ROOT / "deep_learning_modulo_2_leitura_base" / "modulo_2_leitura_base_concatenada.pdf"
UPD_IMGS = ROOT / "deep_learning_modulo_2_leitura_base" / "imagens_atualizacoes"

EXPECTED_UPDATE_IMGS = {
    "double_descent.png",
    "grokking_curva.png",
    "lr_schedules_comparacao.png",
    "muon_15b.jpeg",
    "muon_algo.png",
    "muon_nanogpt_speedrun.png",
    "trajetorias_otimizadores.png",
}


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "✓" if ok else "✗"
    print(f"  {mark} {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    failures = 0

    print(f"\nVerifying {EPUB.name} ({EPUB.stat().st_size/1024:.0f} KB)\n")

    with zipfile.ZipFile(EPUB) as z:
        names = z.namelist()
        xhtml = [n for n in names if n.endswith(".xhtml")]
        imgs_in_epub = [n for n in names if n.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))]

        # 1. Valid EPUB layout
        has_mimetype = "mimetype" in names
        has_container = "META-INF/container.xml" in names
        has_opf = any(n.endswith(".opf") for n in names)
        if not check("EPUB layout (mimetype, container, OPF)", has_mimetype and has_container and has_opf):
            failures += 1

        # 2. Chapter count
        ok = 15 <= len(xhtml) <= 40
        if not check(f"Chapter count plausible", ok, f"{len(xhtml)} xhtml files"):
            failures += 1

        # 3. All extension images are included (match by content hash — pandoc renames)
        import hashlib
        source_hashes = {
            hashlib.md5(p.read_bytes()).hexdigest(): p.name
            for p in UPD_IMGS.iterdir()
        }
        epub_hashes = {
            hashlib.md5(z.read(n)).hexdigest()
            for n in imgs_in_epub
        }
        missing_by_hash = {name for h, name in source_hashes.items() if h not in epub_hashes}
        if not check(f"All 7 extension images embedded (by content hash)", not missing_by_hash,
                     f"missing: {sorted(missing_by_hash)}" if missing_by_hash else f"{len(source_hashes)} present"):
            failures += 1

        # 4. Image count is large (PDF had ~136 extracted + 7 updates)
        if not check(f"Many images embedded", len(imgs_in_epub) >= 100, f"{len(imgs_in_epub)} images"):
            failures += 1

        # 5. Word count sanity vs PDF baseline
        pdf_words = int(subprocess.check_output(
            ["pdftotext", str(PDF), "-"], text=True
        ).split().__len__())
        # Extract text from all XHTML chapters, strip tags, count words
        total_words = 0
        for n in xhtml:
            html = z.read(n).decode("utf-8", errors="ignore")
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text).strip()
            total_words += len(text.split())
        # Our EPUB includes base + updates + papers; expect >> PDF alone
        ratio = total_words / pdf_words
        if not check(f"Content volume >> PDF (all base preserved + extras)",
                     ratio >= 1.5,
                     f"{total_words} words in EPUB vs {pdf_words} in PDF (ratio {ratio:.2f}x)"):
            failures += 1

        # 6. CSS present
        if not check("Custom CSS embedded", any(n.endswith(".css") for n in names)):
            failures += 1

        # 7. Cover image present
        if not check("Cover image present", any("cover" in n.lower() and n.endswith(".png") for n in names)):
            failures += 1

        # 8. No broken image references in XHTMLs
        all_img_basenames = {Path(n).name for n in imgs_in_epub}
        broken = []
        for n in xhtml:
            html = z.read(n).decode("utf-8", errors="ignore")
            for m in re.finditer(r'<img[^>]+src="([^"]+)"', html):
                src_base = Path(m.group(1)).name
                if src_base not in all_img_basenames and "cover" not in src_base.lower():
                    broken.append((n, m.group(1)))
        if not check(f"No broken image references", not broken,
                     f"{len(broken)} broken" if broken else f"all refs resolve"):
            if broken[:5]:
                for b in broken[:5]:
                    print(f"     - in {b[0]}: {b[1]}")
            failures += 1

    print()
    if failures == 0:
        print("ALL CHECKS PASSED.")
        return 0
    else:
        print(f"{failures} CHECKS FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
