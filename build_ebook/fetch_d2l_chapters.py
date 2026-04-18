"""Fetch the 7 base sections directly from d2l.ai and convert to clean markdown.

Rationale: the prior pipeline (`extract_pdf.py` + pymupdf4llm) mangled math and
code in the PDF excerpts of *Dive into Deep Learning*. This replaces that path
by pulling the canonical HTML source (CC BY-SA 4.0) and converting to markdown
with LaTeX math and fenced code blocks preserved.

Inputs : cached HTML under `d2l_cache/` (seeded once via the stdlib fetcher
         here; subsequent builds read the cache and are offline).
Outputs: `base_chapters/ch0{1..7}_*.md` + `d2l_assets/*.{svg,png}`.

The downstream pipeline (`split_updates.py`, `make_cover.py`, `assemble.py`,
`build.sh`) consumes these files unchanged.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "d2l_cache"
ASSETS_SRC = ROOT / "d2l_assets"
CHAP_DIR = ROOT / "base_chapters"
ASSETS_OUT = ROOT / "assets"

CACHE.mkdir(exist_ok=True)
ASSETS_SRC.mkdir(exist_ok=True)
CHAP_DIR.mkdir(exist_ok=True)
ASSETS_OUT.mkdir(exist_ok=True)

BASE = "https://www.d2l.ai"
UPDATE_IMGS = ROOT.parent / "deep_learning_modulo_2_leitura_base" / "imagens_atualizacoes"


@dataclass
class Source:
    cache: str
    url_path: str
    title: str


@dataclass
class Chapter:
    out: str
    heading: str
    sources: list[Source]


CHAPTERS: list[Chapter] = [
    Chapter(
        "ch01_weight_decay.md", "3.7 Weight Decay",
        [Source("ch01_weight_decay.html",
                "/chapter_linear-regression/weight-decay.html",
                "3.7 Weight Decay")],
    ),
    Chapter(
        "ch02_generalization.md", "5.5 Generalization in Deep Learning",
        [Source("ch02_generalization.html",
                "/chapter_multilayer-perceptrons/generalization-deep.html",
                "5.5 Generalization in Deep Learning")],
    ),
    Chapter(
        "ch03_dropout.md", "5.6 Dropout",
        [Source("ch03_dropout.html",
                "/chapter_multilayer-perceptrons/dropout.html",
                "5.6 Dropout")],
    ),
    Chapter(
        "ch04_sgd_minibatch.md", "12.4–12.5 SGD & Minibatch SGD",
        [
            Source("ch04a_sgd.html",
                   "/chapter_optimization/sgd.html",
                   "12.4 Stochastic Gradient Descent"),
            Source("ch04b_minibatch_sgd.html",
                   "/chapter_optimization/minibatch-sgd.html",
                   "12.5 Minibatch Stochastic Gradient Descent"),
        ],
    ),
    Chapter(
        "ch05_momentum.md", "12.6 Momentum",
        [Source("ch05_momentum.html",
                "/chapter_optimization/momentum.html",
                "12.6 Momentum")],
    ),
    Chapter(
        "ch06_adam.md", "12.10 Adam",
        [Source("ch06_adam.html",
                "/chapter_optimization/adam.html",
                "12.10 Adam")],
    ),
    Chapter(
        "ch07_lr_scheduling.md", "12.11 Learning Rate Scheduling",
        [Source("ch07_lr_scheduling.html",
                "/chapter_optimization/lr-scheduler.html",
                "12.11 Learning Rate Scheduling")],
    ),
]


def fetch_html(src: Source) -> str:
    cache_path = CACHE / src.cache
    if not cache_path.exists():
        url = BASE + src.url_path
        print(f"  fetching {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        cache_path.write_bytes(data)
    return cache_path.read_text()


def fetch_image(rel_src: str, page_url_path: str) -> str | None:
    """Download an image referenced from a d2l page; return the local basename.

    `rel_src` is the img[src] attribute (usually "../_images/foo.svg").
    `page_url_path` is the path of the page it was referenced from
    (e.g. "/chapter_optimization/sgd.html"). We resolve the relative URL
    against the absolute page URL so "../_images/foo.svg" becomes
    "https://www.d2l.ai/_images/foo.svg".
    """
    basename = Path(rel_src).name
    local = ASSETS_SRC / basename
    if local.exists() and local.stat().st_size > 0:
        # Sanity: reject cached files that aren't actually image bytes
        head = local.read_bytes()[:64]
        if head.startswith(b"<!DOCTYPE") or head.startswith(b"<html"):
            local.unlink()
        else:
            return basename
    url = urllib.parse.urljoin(BASE + page_url_path, rel_src)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        # Defensive: if d2l served an HTML fallback, treat as failure
        if data[:64].startswith(b"<!DOCTYPE") or data[:64].startswith(b"<html"):
            print(f"  ! image {url} returned HTML, skipping")
            return None
        local.write_bytes(data)
        return basename
    except Exception as e:
        print(f"  ! image fetch failed {url}: {e}")
        return None


# ------------------- HTML cleanup ------------------------------------------
HEADERLINK_RE = re.compile(r"¶")


def clean_section(root, page_url_path: str) -> None:
    """In-place cleanup of the scoped section soup."""
    # Drop nav chrome
    for sel in [
        "div.d2l-tabs",              # Colab/SageMaker link bar
        "a.headerlink",              # ¶ anchors
        "span.eqno",                 # equation numbers "(3.7.1)"
        "div.admonition.seealso",
        "div.admonition.note div.admonition-title a",
    ]:
        for n in root.select(sel):
            n.decompose()

    # Strip site-logo and external images (keep only ../_images/ content)
    for img in root.find_all("img"):
        src = img.get("src", "")
        if "_images/" not in src:
            img.decompose()

    # Collapse code-framework tab panels to PyTorch-only
    # <div class="mdl-tabs"> contains <div class="mdl-tabs__panel" id="pytorch-X">
    for tabs in root.select("div.mdl-tabs"):
        panels = tabs.select("div.mdl-tabs__panel")
        pytorch = next((p for p in panels if (p.get("id") or "").startswith("pytorch")), None)
        # Fallback: first panel
        keep = pytorch or (panels[0] if panels else None)
        if keep is None:
            tabs.decompose()
            continue
        # Unwrap: replace the whole tabs div with the kept panel's children
        parent = tabs.parent
        idx = list(parent.contents).index(tabs)
        # Extract keep's children
        kept_nodes = list(keep.children)
        tabs.extract()
        for offset, node in enumerate(kept_nodes):
            parent.insert(idx + offset, node.extract() if hasattr(node, "extract") else node)

    # Convert inline math: <span class="math notranslate nohighlight">\(...\)</span>
    for span in root.select("span.math"):
        tex = span.get_text()
        tex = re.sub(r"^\\\(", "", tex)
        tex = re.sub(r"\\\)$", "", tex)
        tex = tex.strip()
        span.replace_with(NavigableString(f"${tex}$"))

    # Convert display math: <div class="math ...">\[...\]</div>
    for div in root.find_all("div", class_="math"):
        tex = div.get_text()
        tex = re.sub(r"^\s*\\\[", "", tex)
        tex = re.sub(r"\\\]\s*$", "", tex)
        # Strip any leftover eqno text like "(3.7.1)" at start (if span.eqno survived)
        tex = re.sub(r"^\s*\([\d.]+\)\s*", "", tex)
        tex = tex.strip()
        # Emit as block-level math; surrounding newlines matter for pandoc
        replacement_html = f"\n\n$$\n{tex}\n$$\n\n"
        div.replace_with(NavigableString(replacement_html))

    # Convert code blocks: <div class="highlight-python"> ... <pre>...</pre>
    # Grab plain text (strips pygments spans), wrap in fenced block.
    for hl in list(root.select("div.highlight-python, div.highlight-default")):
        pre = hl.find("pre")
        if pre is None:
            continue
        code = pre.get_text()
        # Normalize trailing newlines
        code = code.rstrip() + "\n"
        # Wrap in fenced block; inject a raw HTML comment wrapper so pandoc
        # treats this as a preformatted literal (use a <pre><code>).
        fenced = BeautifulSoup(
            f"<pre><code class=\"language-python\">{escape_html(code)}</code></pre>",
            "lxml",
        )
        # lxml wraps in html/body; extract body's children
        body = fenced.body
        repl = list(body.children)
        parent = hl.parent
        idx = list(parent.contents).index(hl)
        hl.extract()
        for offset, node in enumerate(repl):
            parent.insert(idx + offset, node.extract())

    # Output-cells rendered as <div class="output_area"><pre>...</pre></div> → plain pre/code
    for out in list(root.select("div.output_area, div.output")):
        pre = out.find("pre")
        if pre:
            code = pre.get_text().rstrip() + "\n"
            if not code.strip():
                out.decompose()
                continue
            fenced = BeautifulSoup(
                f"<pre><code class=\"language-text\">{escape_html(code)}</code></pre>",
                "lxml",
            )
            body = fenced.body
            repl = list(body.children)
            parent = out.parent
            idx = list(parent.contents).index(out)
            out.extract()
            for offset, node in enumerate(repl):
                parent.insert(idx + offset, node.extract())
        else:
            out.decompose()

    # Download referenced images and rewrite src attributes; clear alt (d2l
    # uses the src path as alt text, which is useless and ugly when escaped).
    for img in root.find_all("img"):
        src = img.get("src", "")
        if not src:
            continue
        basename = fetch_image(src, page_url_path)
        if basename:
            img["src"] = f"assets/{basename}"
            img["alt"] = ""
        else:
            img.decompose()

    # Drop citation/reference links entirely — d2l renders them as nested
    # tooltip <a>...(<cite>author</cite> (<span>year</span>))...</a>, which
    # pandoc flattens into very long title-attribute chains. The updates
    # chapter and References appendix already cover canonical citations.
    for a in list(root.select("a.reference.internal")):
        # Only keep if it targets an http(s) URL (external legit link)
        href = a.get("href", "")
        if href.startswith(("http://", "https://")):
            continue
        text = a.get_text(" ", strip=True)
        a.replace_with(NavigableString(text))

    # Remove leftover headerlink text ("¶" artifact inside headings)
    for h in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        # Strip leading section number prefix like "3.7." or "3.7.3."
        text = h.get_text()
        # Also strip the trailing pilcrow if somehow it survived
        text = text.replace("¶", "")
        # Normalize: "3.7.Weight Decay" -> "3.7 Weight Decay"
        text = re.sub(r"^(\d+(?:\.\d+)+)\.", r"\1 ", text)
        # Replace the children with a single clean text node
        h.clear()
        h.append(NavigableString(text.strip()))


def escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def extract_section(html: str, page_url_path: str) -> str:
    """Return cleaned HTML of the main chapter section."""
    soup = BeautifulSoup(html, "lxml")
    # Page-content may contain multiple section divs; the first with an id is the chapter root.
    root = soup.select_one("div.document div.page-content div.section[id]")
    if root is None:
        raise RuntimeError("could not locate main section div")
    clean_section(root, page_url_path)
    return str(root)


def html_to_markdown(html: str) -> str:
    """Shell out to pandoc to get pandoc-markdown with `$...$` math preserved."""
    result = subprocess.run(
        [
            "pandoc",
            "-f", "html+tex_math_dollars",
            "-t", "markdown+tex_math_dollars-raw_html-raw_attribute-fenced_divs-bracketed_spans-native_divs-native_spans",
            "--wrap=none",
            "--markdown-headings=atx",
        ],
        input=html,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout


# ------------------- final markdown polish ---------------------------------
HEADER_LEAK_RE = re.compile(
    r"^(Weight Decay|Dropout|Generalization(?: in Deep Learning)?|Momentum|Adam|"
    r"Stochastic Gradient Descent|Minibatch Stochastic Gradient Descent|"
    r"Learning Rate Scheduling)\s*$",
    re.MULTILINE,
)

REFMARK_RE = re.compile(r"(?<!\w)\[\d{2,4}\](?!\()")


def polish_markdown(md: str, top_heading: str) -> str:
    # Prepend our canonical H1
    md = f"# {top_heading}\n\n" + md.strip() + "\n"
    # Strip reference-marker leaks like "[169]" or "[84]"
    md = REFMARK_RE.sub("", md)
    # Strip page-header leaks
    md = HEADER_LEAK_RE.sub("", md)
    # Collapse 3+ blank lines
    md = re.sub(r"\n{3,}", "\n\n", md)
    # Strip pandoc attribute annotations on links like {.reference .internal}
    md = re.sub(r"\]\(([^)]+)\)\{[^}]*\}", r"](\1)", md)
    # Strip attribute annotations on inline code/spans like `x`{.foo}
    md = re.sub(r"`\{[^}]*\}", "`", md)
    # Strip fenced-block info-annotations like ``` {.python .sourceCode}
    md = re.sub(r"^``` \{[^}]*\}$", "```", md, flags=re.MULTILINE)
    # Rewrite d2l cross-refs to plain text (target HTMLs don't exist in EPUB)
    md = re.sub(
        r"\[([^\]]+)\]\([^)]*\.html[^)]*\)",
        r"*\1*",
        md,
    )
    # Defensive: any leftover d2l anchor-only links
    md = re.sub(r"\[([^\]]+)\]\(#[^)]+\)", r"*\1*", md)
    return md.strip() + "\n"


# ------------------- orchestration -----------------------------------------

def build_chapter(ch: Chapter) -> None:
    section_mds: list[str] = []
    multi = len(ch.sources) > 1
    for src in ch.sources:
        html = fetch_html(src)
        cleaned = extract_section(html, src.url_path)
        md = html_to_markdown(cleaned)
        # Demote pandoc's own H1 (from the page's section). When concatenating
        # two sources (e.g. §12.4 + §12.5), keep it as H2 with the section
        # title; when there's a single source, drop it and let the chapter H1
        # from polish_markdown carry the title.
        if multi:
            md = re.sub(r"^# .*\n+", f"## {src.title}\n\n", md,
                        count=1, flags=re.MULTILINE)
        else:
            md = re.sub(r"^# .*\n+", "", md, count=1, flags=re.MULTILINE)
        section_mds.append(md)

    combined = "\n\n".join(s.strip() for s in section_mds)
    final = polish_markdown(combined, ch.heading)
    (CHAP_DIR / ch.out).write_text(final)
    words = len(final.split())
    print(f"  wrote {ch.out}: {words} words")


def sync_assets() -> None:
    """Mirror d2l_assets/ and imagens_atualizacoes/ into assets/ for pandoc."""
    for img in ASSETS_SRC.iterdir():
        dst = ASSETS_OUT / img.name
        if not dst.exists() or dst.stat().st_mtime < img.stat().st_mtime:
            shutil.copy2(img, dst)
    for img in UPDATE_IMGS.iterdir():
        dst = ASSETS_OUT / img.name
        if not dst.exists() or dst.stat().st_mtime < img.stat().st_mtime:
            shutil.copy2(img, dst)
    print(f"  assets/: {len(list(ASSETS_OUT.glob('*')))} files")


def main() -> None:
    # Seed cache if missing (one-time online requirement)
    for ch in CHAPTERS:
        for src in ch.sources:
            fetch_html(src)
    # Build chapters
    for ch in CHAPTERS:
        print(f"building {ch.out} ← {[s.url_path for s in ch.sources]}")
        build_chapter(ch)
    # Sync images for pandoc's --resource-path
    sync_assets()


if __name__ == "__main__":
    main()
