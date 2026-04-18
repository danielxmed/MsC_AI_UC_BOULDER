"""Split the updates markdown into per-chapter fragments aligned with the base PDF sections.

Mapping:
  ## 2. Weight Decay            → ch01 (§3.7)
  ## 3. Generalização           → ch02 (§5.5)
  ## 4. Dropout                 → ch03 (§5.6)
  ## 5. SGD e Minibatch         → ch04 (§12.4-12.5)
  ## 6. Momentum                → ch05 (§12.6)
  ## 7. Adam → AdamW            → ch06 (§12.10)
  ## 9. Learning-Rate Scheduling→ ch07 (§12.11)

Sections 1 (executive summary), 8 (new optimizers), 10 (meta-lessons), 11 (summary table),
12 (references), Appendix A (trajectories) — routed to frontmatter / backmatter.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPDATE_MD = ROOT / "deep_learning_modulo_2_leitura_base" / "atualizacoes_otimizacao_2023_2026.md"
OUT_DIR = Path(__file__).resolve().parent / "updates"
OUT_DIR.mkdir(exist_ok=True)

TARGETS = [
    # (out_name, regex for heading start, optional regex for heading end)
    ("intro.md",           r"^## 1\. Sumário executivo", r"^## 2\."),
    ("update_ch01.md",     r"^## 2\. Weight Decay",      r"^## 3\."),
    ("update_ch02.md",     r"^## 3\. Generalização",     r"^## 4\."),
    ("update_ch03.md",     r"^## 4\. Dropout",           r"^## 5\."),
    ("update_ch04.md",     r"^## 5\. SGD",               r"^## 6\."),
    ("update_ch05.md",     r"^## 6\. Momentum",          r"^## 7\."),
    ("update_ch06.md",     r"^## 7\. Adam",              r"^## 8\."),
    ("new_optimizers.md",  r"^## 8\. Novos otimizadores",r"^## 9\."),
    ("update_ch07.md",     r"^## 9\. Learning-Rate",     r"^## 10\."),
    ("meta_lessons.md",    r"^## 10\. Meta-lições",      r"^## 11\."),
    ("summary_table.md",   r"^## 11\. Tabela-resumo",    r"^## 12\."),
    ("references.md",      r"^## 12\. Referências",      r"^## Apêndice A"),
    ("appendix_A.md",      r"^## Apêndice A",            None),
]


def rewrite_images(md: str) -> str:
    # imagens_atualizacoes/foo.png  → assets/foo.png
    return re.sub(
        r"imagens_atualizacoes/",
        "assets/",
        md,
    )


def main() -> None:
    text = UPDATE_MD.read_text()

    for name, start_re, end_re in TARGETS:
        m_start = re.search(start_re, text, re.MULTILINE)
        if not m_start:
            print(f"  ! start not found for {name}")
            continue
        start = m_start.start()
        if end_re:
            m_end = re.search(end_re, text, re.MULTILINE)
            end = m_end.start() if m_end else len(text)
        else:
            end = len(text)
        fragment = text[start:end].rstrip() + "\n"
        fragment = rewrite_images(fragment)
        # also drop horizontal rules and trailing italics footer
        fragment = re.sub(r"^---\s*$", "", fragment, flags=re.MULTILINE)
        fragment = re.sub(r"\n{3,}", "\n\n", fragment).strip() + "\n"
        (OUT_DIR / name).write_text(fragment)
        print(f"  wrote {name}: {len(fragment.split())} words")


if __name__ == "__main__":
    main()
