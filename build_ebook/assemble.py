"""Assemble all pieces into a single unified markdown for pandoc.

Order:
  Frontmatter → Preface → Intro (executive summary from updates)
  Part I (7 chapters, base + update box per chapter)
  Part II (10 paper deep-dives)
  Backmatter (meta-lessons, summary table, decision map, references, appendix A)
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHAP = ROOT / "base_chapters"
UPD = ROOT / "updates"
PAP = ROOT / "papers"
OUT = ROOT / "ebook.md"

# ---------- helpers ----------
def read(p: Path) -> str:
    return p.read_text().strip() + "\n"


def shift_headings(md: str, by: int) -> str:
    """Increase heading levels by `by` (#, ##, ... up to 6)."""
    def repl(m: re.Match) -> str:
        hashes = m.group(1)
        rest = m.group(2)
        new_level = min(len(hashes) + by, 6)
        return "#" * new_level + rest
    return re.sub(r"^(#{1,6})(\s.*)$", repl, md, flags=re.MULTILINE)


def wrap_update(md: str) -> str:
    """Wrap markdown in a fenced div with class update-box."""
    return ":::: update-box\n\n" + md.strip() + "\n\n::::\n"


# ---------- frontmatter ----------
FRONTMATTER = r"""---
title: "Otimização em Deep Learning"
subtitle: "Módulo 2 — Base (Dive into Deep Learning) + Atualizações 2023→2026"
author: "Compilado para o MsC AI · UC Boulder"
lang: pt-BR
date: "Abril de 2026"
rights: "Excertos do livro *Dive into Deep Learning* (Zhang, Lipton, Li, Smola) sob licença CC BY-SA 4.0. Extensões e comentários editoriais são originais deste ebook."
cover-image: cover.png
---

# Prefácio do editor {-}

Este ebook é uma **edição didática unificada** do material do Módulo 2 da pós-graduação em
Inteligência Artificial (MsC AI, UC Boulder). Ele integra, num único arquivo autocontido:

1. **O conteúdo integral do PDF de leitura-base** — os excertos do livro *Dive into Deep Learning*
   (Zhang, Lipton, Li, Smola) que cobrem §3.7, §5.5, §5.6, §12.4–12.6, §12.10 e §12.11. Preservados
   em inglês, tais como publicados pelos autores sob licença CC BY-SA 4.0.
2. **Extensões 2023→2026 em português** — um dossiê escrito para atualizar o livro à prática de
   ponta em 2026, apontando o que continua válido, o que está parcialmente desatualizado e o que
   é legado.
3. **Deep-dives em papers selecionados** — resumos substanciais de dez trabalhos centrais para
   entender otimização moderna: de AdamW (2019) a Muon (2024-2025), passando por Schedule-Free,
   Chinchilla, Grokking e Double Descent.

A ordem de leitura sugerida é **linear**: cada capítulo do livro-base é seguido imediatamente por
um bloco *"O que mudou desde 2023"*, marcado em azul na renderização. Para quem quer uma referência
rápida, a Parte II oferece os capítulos de paper isolados, que podem ser lidos sem sequência
obrigatória. O apêndice traz um mapa de decisão e um glossário bilíngue.

**Sobre a linguagem**. O texto do livro-base está em inglês (tradução acarretaria risco desnecessário
de erro em conteúdo técnico denso). Prefácio, extensões, deep-dives e notas editoriais estão em
português. Símbolos matemáticos seguem a convenção internacional em ambos.

**Sobre imagens**. Todas as figuras do PDF-base e as sete figuras específicas das extensões
(double descent, grokking, trajetórias de otimizadores, comparação de schedules, algoritmo do
Muon, benchmarks do Muon/NanoGPT, Muon a 1.5B) estão embutidas neste arquivo.

Boa leitura.

---
"""

# ---------- Part I definitions ----------
PART_I_HEAD = """
# Parte I — Fundamentos: o PDF-base e suas atualizações {-}

A Parte I é a espinha dorsal do módulo. Sete capítulos reproduzem integralmente os
excertos do livro *Dive into Deep Learning* (2021/2023) que compõem a leitura-base.
Cada um termina com um bloco destacado — **O que mudou desde 2023** — que atualiza o
conteúdo à prática de 2026 e aponta os papers relevantes, detalhados na Parte II.

"""

PART_I = [
    ("ch01_weight_decay",   "update_ch01", "Capítulo 1 — §3.7 Weight Decay"),
    ("ch02_generalization", "update_ch02", "Capítulo 2 — §5.5 Generalization in Deep Learning"),
    ("ch03_dropout",        "update_ch03", "Capítulo 3 — §5.6 Dropout"),
    ("ch04_sgd_minibatch",  "update_ch04", "Capítulo 4 — §12.4 e §12.5 SGD e Minibatch SGD"),
    ("ch05_momentum",       "update_ch05", "Capítulo 5 — §12.6 Momentum"),
    ("ch06_adam",           "update_ch06", "Capítulo 6 — §12.10 Adam"),
    ("ch07_lr_scheduling",  "update_ch07", "Capítulo 7 — §12.11 Learning Rate Scheduling"),
]

# ---------- Part II definitions ----------
PART_II_HEAD = """
# Parte II — Deep-dives nos papers pós-2023 {-}

A Parte II aprofunda dez trabalhos centrais para entender o estado da arte em 2026.
Os capítulos podem ser lidos em qualquer ordem; referenciados um a um nos boxes de
atualização da Parte I.

"""

PART_II = [
    "paper01_adamw",
    "paper02_why_weight_decay",
    "paper03_lion",
    "paper04_sophia",
    "paper05_muon",
    "paper06_schedule_free",
    "paper07_chinchilla",
    "paper08_grokking",
    "paper09_double_descent",
    "paper10_wsd_d2z",
]

# ---------- Backmatter ----------
BACK_HEAD = """
# Parte III — Síntese e apêndices {-}
"""


def chapter_block(base_slug: str, update_slug: str, title: str) -> str:
    """Build a single Part I chapter: heading + base content + update box."""
    base_md = read(CHAP / f"{base_slug}.md")
    upd_md = read(UPD / f"{update_slug}.md")

    # Strip the leading H1 from base chapter and use our own title
    base_md = re.sub(r"^#\s.*$\n+", "", base_md, count=1, flags=re.MULTILINE)

    # Shift base headings down by 1 (so chapter title stays H1)
    base_md = shift_headings(base_md, 1)

    # Normalize updates: its markdown has ## 2. Weight Decay etc.
    # Strip its leading H2 section number, we'll provide a clean H2 "Atualizações 2023→2026"
    upd_md = re.sub(r"^##\s+\d+\.\s+.*$\n+", "", upd_md, count=1, flags=re.MULTILINE)
    upd_md = shift_headings(upd_md, 1)  # shift inner H2/H3/H4 down

    chapter = [
        f"# {title}",
        "",
        "## Texto-base (Dive into Deep Learning) {-}",
        "",
        base_md.strip(),
        "",
        wrap_update(
            "## O que mudou desde 2023 — atualização editorial\n\n" + upd_md.strip()
        ),
        "",
    ]
    return "\n".join(chapter)


def paper_block(slug: str) -> str:
    md = read(PAP / f"{slug}.md")
    # Convert leading "# Paper N — title" to a top-level H1 under Part II
    md = re.sub(r"^#\s+Paper\s+\d+\s*—\s*", "# ", md, count=1, flags=re.MULTILINE)
    return md.strip() + "\n"


def main() -> None:
    parts: list[str] = [FRONTMATTER.strip(), ""]

    # Executive intro from updates
    intro_md = read(UPD / "intro.md")
    intro_md = re.sub(r"^##\s+1\.\s+.*$\n+", "", intro_md, count=1, flags=re.MULTILINE)
    parts.append("# Sumário executivo: legado × prática 2026 {-}")
    parts.append("")
    parts.append(intro_md.strip())
    parts.append("")

    # Part I
    parts.append(PART_I_HEAD.strip())
    parts.append("")
    for base_slug, upd_slug, title in PART_I:
        parts.append(chapter_block(base_slug, upd_slug, title))
        parts.append("")

    # Residual fragments from the PDF that crossed section boundaries
    frag = CHAP / "ch08_fragments.md"
    if frag.exists():
        frag_md = read(frag)
        frag_md = re.sub(r"^#\s.*$\n+", "", frag_md, count=1, flags=re.MULTILINE)
        frag_md = shift_headings(frag_md, 1)
        parts.append("# Capítulo 8 — Trechos complementares do PDF-base")
        parts.append("")
        parts.append(frag_md.strip())
        parts.append("")

    # Part II
    parts.append(PART_II_HEAD.strip())
    parts.append("")
    for slug in PART_II:
        parts.append(paper_block(slug))
        parts.append("")

    # Catalogue of other optimizers (from the updates new_optimizers.md section)
    new_opt = read(UPD / "new_optimizers.md")
    new_opt = re.sub(r"^##\s+8\.\s+", "# Capítulo 18 — Catálogo dos demais otimizadores pós-2023\n\n## ", new_opt, count=1, flags=re.MULTILINE)
    parts.append(new_opt.strip())
    parts.append("")

    # Backmatter: Part III
    parts.append(BACK_HEAD.strip())
    parts.append("")

    # Meta-lessons
    meta = read(UPD / "meta_lessons.md")
    meta = re.sub(r"^##\s+10\.\s+Meta-lições[^\n]*$", "# Capítulo 19 — Meta-lições: scaling laws, precisão e LLMs", meta, count=1, flags=re.MULTILINE)
    parts.append(meta.strip())
    parts.append("")

    # Summary table
    table = read(UPD / "summary_table.md")
    table = re.sub(r"^##\s+11\.\s+[^\n]*$", "# Capítulo 20 — Tabela-resumo: legado × prática 2026", table, count=1, flags=re.MULTILINE)
    parts.append(table.strip())
    parts.append("")

    # Appendix A
    app_a = read(UPD / "appendix_A.md")
    app_a = re.sub(r"^##\s+Apêndice A[^\n]*$", "# Apêndice A — Trajetórias ilustrativas (SGD, Momentum, Adam)", app_a, count=1, flags=re.MULTILINE)
    parts.append(app_a.strip())
    parts.append("")

    # Decision map (handwritten)
    parts.append("# Apêndice B — Mapa de decisão de otimizadores (2026)")
    parts.append("""
Um guia rápido para escolher o otimizador conforme o regime de treino.

| Regime | Recomendação 2026 | Alternativa de pesquisa |
|---|---|---|
| CV / ResNet / ViT pequeno-médio (multi-época) | **SGD + momentum + cosine** | Lion |
| Difusão / Stable-Diffusion-class | **Lion** | AdamW |
| LLM fine-tuning em dataset pequeno | **AdamW** (seguro) | Prodigy (parameter-free) |
| LLM pretraining < 1B parâmetros | **AdamW** ou Schedule-Free AdamW | Lion, Sophia |
| LLM pretraining ≥ 1B parâmetros | **AdamW** (produção) | **Muon** (fronteira) |
| Infraestrutura avançada, equipe grande | **SOAP / DistributedShampoo** | Muon + Shampoo híbrido |

Regras transversais aplicáveis em 2026:

- **AdamW, não Adam.** Sempre decoupled weight decay.
- **Warmup sempre**: 1–10% dos passos, linear de 0 a $\\eta_{\\max}$.
- **Gradient clipping**: `clip_grad_norm_` com `max_norm=1.0`.
- **bf16 default**: precisão mista é padrão; $\\epsilon$ do Adam passa para 1e-7.
- **Weight decay $\\lambda \\in [0.01, 0.1]$**, excluindo biases e ganhos de normalização.
- **Schedule**: cosine-a-10% continua aceitável, mas WSD/D2Z/Schedule-Free são superiores.
""")
    parts.append("")

    # Glossary
    parts.append("# Apêndice C — Glossário bilíngue")
    parts.append("""
Termos-chave ordenados alfabeticamente, com tradução e equivalente técnico.

- **AdamW** — Adam com *decoupled weight decay*. Default para transformers desde ~2019.
- **bfloat16 / bf16** — Formato de ponto flutuante de 16 bits com 8 bits de expoente e 7 de mantissa; default em LLM training por ter intervalo dinâmico amplo.
- **bias-variance trade-off** — Compromisso viés-variância; sob interpretação clássica, uma curva em U entre complexidade do modelo e erro de teste. Substituído por *double descent* no regime sobreparametrizado.
- **Chinchilla-optimal** — Regime compute-optimal de Hoffmann et al. (2022) onde parâmetros e tokens escalam na mesma proporção (~20 tokens/parâmetro). Hoje superado por regimes *inference-aware* com muito mais tokens por parâmetro.
- **D2Z (Decay-to-Zero)** — Schedule linear que decai a LR até zero; Bergsma et al. 2025.
- **Decoupled weight decay** — Decaimento de pesos aplicado *fora* do passo adaptativo (a diferença AdamW × Adam).
- **Double descent** — Curva não-monotônica de test error × capacidade do modelo, com pico no threshold de interpolação e segunda descida em regime sobreparametrizado.
- **DropPath / Stochastic Depth** — Variante estruturada de dropout que "droppa" blocos residuais inteiros; padrão em ViTs.
- **Edge of Stability (EoS)** — Regime em que o maior autovalor do Hessiano oscila perto de $2/\\eta$; não converge no sentido clássico mas generaliza bem.
- **Grokking** — Fenômeno de generalização tardia abrupta após plateau longo em validation.
- **Lion (EvoLved Sign Momentum)** — Otimizador descoberto por busca simbólica; Chen et al. 2023.
- **Muon (Momentum Orthogonalized)** — Otimizador que aplica Newton-Schulz para ortogonalizar o momentum antes do update; usado no Kimi K2 trilionário.
- **Newton-Schulz** — Iteração polinomial para aproximar a matriz ortogonal mais próxima; estável em bf16.
- **Schedule-Free** — Classe de otimizadores que elimina a necessidade de schedule de LR via Polyak-Ruppert averaging.
- **Sophia** — Otimizador de segunda ordem que usa estimativa diagonal da Hessiana.
- **SAM (Sharpness-Aware Minimization)** — Otimizador que minimiza a loss na vizinhança dos parâmetros para buscar mínimos planos.
- **Scaling laws** — Leis empíricas que relacionam loss a compute, parâmetros e dados.
- **Warmup** — Fase inicial de aumento linear da LR de 0 a $\\eta_{\\max}$, universal em LLMs.
- **WSD (Warmup-Stable-Decay)** — Schedule de três fases popularizado em 2024.
""")
    parts.append("")

    # References
    refs = read(UPD / "references.md")
    refs = re.sub(r"^##\s+12\.\s+Referências[^\n]*$", "# Referências completas", refs, count=1, flags=re.MULTILINE)
    parts.append(refs.strip())
    parts.append("")

    # Final colophon
    parts.append("""
# Colofão {-}

Este ebook foi compilado em abril de 2026 a partir de três fontes:

1. `modulo_2_leitura_base_concatenada.pdf` — excertos do livro *Dive into Deep Learning*
   (Zhang, Lipton, Li, Smola), disponível em [d2l.ai](https://d2l.ai) sob licença
   CC BY-SA 4.0.
2. `atualizacoes_otimizacao_2023_2026.md` — documento editorial em português preparado
   para o MsC AI · UC Boulder.
3. Conteúdo dos papers da Parte II, resumido pelo editor a partir das fontes primárias
   linkadas em cada capítulo.

A tipografia é Charter (corpo), Inter (títulos) e Fira Code (código). As equações
renderizam em MathML para compatibilidade com leitores modernos. O CSS foi ajustado
para legibilidade prolongada, com variante dark-mode automática em leitores que
honram `prefers-color-scheme`.

*Boa leitura — e bom estudo.*
""")

    # Write
    OUT.write_text("\n".join(parts) + "\n")
    wc = len(OUT.read_text().split())
    print(f"wrote {OUT.name}: {wc} words")


if __name__ == "__main__":
    main()
