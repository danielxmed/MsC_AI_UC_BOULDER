"""Post-process the EPUB produced by pandoc.

Pandoc's `epub3` writer renames every bundled `.svg` file to `.svgz`, even
when the underlying bytes are not gzipped. Some readers (notably Apple
Books) refuse to render `.svgz` without the corresponding Content-Encoding
header, which an EPUB ZIP cannot provide. This script:

  1. Opens the EPUB.
  2. Scans the OPF manifest and every XHTML for `.svgz` references whose
     content is actually uncompressed SVG (starts with `<?xml` or `<svg`).
  3. Rewrites those references to `.svg` and renames the files inside the
     ZIP accordingly.

Works in-place on the passed path.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


def is_uncompressed_svg(data: bytes) -> bool:
    head = data[:80].lstrip()
    return head.startswith(b"<?xml") or head.startswith(b"<svg")


def main(path: str) -> int:
    epub_path = Path(path)
    if not epub_path.exists():
        print(f"!! {epub_path} not found")
        return 1

    tmp = tempfile.NamedTemporaryFile(suffix=".epub", delete=False)
    tmp.close()
    tmp_path = Path(tmp.name)

    rename_map: dict[str, str] = {}
    with zipfile.ZipFile(epub_path) as zin:
        # First pass: identify svgz entries that are really plain SVG
        for info in zin.infolist():
            if info.filename.lower().endswith(".svgz"):
                data = zin.read(info.filename)
                if is_uncompressed_svg(data):
                    rename_map[info.filename] = info.filename[:-5] + ".svg"

        if not rename_map:
            print("  post_pandoc_epub: no .svgz → .svg renames needed")
            tmp_path.unlink()
            return 0

        # Build basename rewrite rules to patch references
        basename_rewrite = {
            Path(k).name: Path(v).name for k, v in rename_map.items()
        }

        # Second pass: rewrite archive contents
        with zipfile.ZipFile(
            tmp_path, "w",
            compression=zipfile.ZIP_DEFLATED,
            allowZip64=True,
        ) as zout:
            for info in zin.infolist():
                data = zin.read(info.filename)
                new_name = rename_map.get(info.filename, info.filename)

                # Rewrite references in text-like files
                if info.filename.lower().endswith(
                    (".xhtml", ".html", ".opf", ".ncx", ".css")
                ):
                    txt = data.decode("utf-8", errors="ignore")
                    for old, new in basename_rewrite.items():
                        txt = txt.replace(old, new)
                    # Content-Type overrides in the OPF manifest: pandoc
                    # writes `image/svg+xml` for both; nothing to fix.
                    data = txt.encode("utf-8")

                # "mimetype" must be stored first and uncompressed
                if info.filename == "mimetype":
                    new_info = zipfile.ZipInfo("mimetype")
                    new_info.compress_type = zipfile.ZIP_STORED
                    zout.writestr(new_info, data)
                else:
                    new_info = zipfile.ZipInfo(new_name)
                    new_info.compress_type = info.compress_type
                    new_info.date_time = info.date_time
                    zout.writestr(new_info, data)

    shutil.move(str(tmp_path), str(epub_path))
    print(f"  post_pandoc_epub: renamed {len(rename_map)} .svgz → .svg")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else
                  "deep_learning_modulo_2_ebook.epub"))
