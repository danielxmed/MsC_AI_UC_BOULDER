# Paper 10 — Learning-Rate Schedules: WSD e D2Z

> **Hägele, Bach, Jaggi, Flammarion et al. 2024. *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations.*** [arXiv:2410.05192](https://arxiv.org/html/2410.05192v1)
>
> **Bergsma, Guan, Ungar 2025. *Why Linearly Decaying the Learning Rate to Zero Works Best.*** [arXiv:2502.15938](https://arxiv.org/pdf/2502.15938)

## Contexto

O §12.11 do D2L termina o capítulo 12 apresentando cosine e decay polinomial como schedules principais. Em 2024-2025, duas linhas de pesquisa atualizaram esse cenário: **WSD** (Warmup-Stable-Decay) e **D2Z** (Decay-to-Zero linear) — ambas superam cosine-a-10% em pretraining de LLMs e adicionam flexibilidade operacional. Este capítulo cobre os dois papers em conjunto, pois se complementam.

## WSD — a filosofia das três fases

A agenda Warmup-Stable-Decay tem três fases:

1. **Warmup linear**: LR sobe de 0 a $\eta_{\max}$ em 1-2% dos passos totais.
2. **Stable**: LR fica em $\eta_{\max}$ por ~80-90% dos passos. Sem decaimento.
3. **Decay final**: LR despenca rapidamente (linear, exponencial ou cosine) até perto de 0 nos últimos 10-20% dos passos.

A vantagem operacional em relação a cosine é imediata: **você pode parar antes da fase decay, congelar um checkpoint, e continuar treinando depois estendendo a fase stable**. Com cosine, o decaimento começa desde cedo e estender o treino implica mudar o schedule inteiro (há um "horizon lock").

## A paisagem de vale de rio

Hägele et al. explicam empiricamente **por que WSD funciona** via a metáfora do *river-valley loss landscape*: a superfície de loss tem um vale profundo (o "rio") ao longo do qual estão os bons mínimos, mas com vales laterais rasos em torno.

- Durante a fase stable, com LR alta, o otimizador **oscila no vale** — a loss fica elevada mas o modelo *move-se rapidamente ao longo do rio*, explorando a estrutura do subespaço bom.
- No decay final, as oscilações colapsam e o modelo **afunda** no fundo do rio — a loss cai abruptamente e frequentemente termina **abaixo** do equivalente cosine-a-10%.

Isto explica a observação contraintuitiva de que durante a fase stable a loss parece estagnada, mas o decay no final é muito mais produtivo do que seria esperável de uma queda comparável em cosine.

## WSD-S: reaproveitamento de decays

Uma variante importante — **WSD-S** — adiciona um truque: quando você precisa de um checkpoint a um budget intermediário, pode ramificar (branch) do ponto de stable e decair brevemente. O "ramo principal" continua em stable indefinidamente. Múltiplos checkpoints em múltiplos budgets de compute, em uma única run.

Empiricamente: "WSD-S supera WSD e Cyclic-Cosine ao obter múltiplos checkpoints de modelos de linguagem em vários orçamentos computacionais em uma única execução para parâmetros variando de 0.1B a 1.2B" (Hägele et al.).

## D2Z — o decaimento linear para zero

Bergsma et al. (2025) estudam uma variação diferente: substituir o "floor" do cosine (que para em ~0.1 $\eta_{\max}$) por **zero verdadeiro, alcançado linearmente**. O resultado é que D2Z supera cosine-a-10% em praticamente todo regime de LLM pretraining, com margem que cresce com tokens-per-parameter.

A interpretação teórica dos autores é elegante: AdamW pode ser lido como **média móvel exponencial dos updates de peso**. O LR no fim do treino controla a fração de "últimos updates" que permanecem na média final. Se o LR é $\approx 0$, a média reflete o estado *estável* atingido com LR alto. Se é não-zero, há contaminação pelo ruído recente.

Evidência empírica:

- 610M parâmetros treinados com **D2Z até 80 tokens/parâmetro** ≈ loss equivalente a cosine até **200 tokens/parâmetro** — **economia de 60% em compute**.
- Extrapolação: LLaMA 2 7B (treinado em 286 tokens/parâmetro com cosine) poderia ter atingido mesma loss em ~114 tokens/parâmetro com D2Z.

A economia é substancial quando o orçamento total já é de trilhões de tokens.

## Comparação WSD × D2Z × cosine × Schedule-Free

| Schedule | Precisa saber $T$? | LR ao final | Vence cosine-10%? | Comentário |
|---|---|---|---|---|
| Cosine to 10% | Sim | $0.1\,\eta_{\max}$ | Baseline | Dominante 2018-2023 |
| WSD | Não (pode estender stable) | Próximo de 0 | Sim | Flexível operacionalmente |
| D2Z linear | Sim | 0 | Sim, com margem | Mais simples que WSD, melhor que cosine |
| Schedule-Free | Não | N/A (sem schedule) | Comparável | Venceu MLCommons 2024 |

## Recomendações em 2026

- Se você sabe $T$ com precisão e quer simplicidade: **D2Z linear**.
- Se você tem budget flexível ou quer múltiplos checkpoints: **WSD-S**.
- Se você quer zero tuning: **Schedule-Free AdamW** (Paper 6).
- Cosine-a-10% continua sendo o baseline aceitável, mas **não é mais o estado da arte**.

## Um detalhe sobre warmup

Todos os três schedules modernos (WSD, D2Z, cosine) começam com warmup linear. **Warmup é universal em LLMs** — absolutamente nenhum transformer de frontier em 2023-2026 dispensa. O livro original não menciona warmup; considere isto a lacuna mais gritante do §12.11. Duração típica: 1-10% dos passos totais, linear de 0 a $\eta_{\max}$.

## Status em 2026

WSD e D2Z, somados ao Schedule-Free, representam a atualização completa do §12.11. Espera-se que no próximo ciclo (2026-2027) o consenso em produção mude de cosine para D2Z como default, com WSD-S em pipelines que precisam de múltiplos checkpoints. Conhecer os três é essencial para quem ler papers ou código de 2024 em diante.
