# Paper 7 — Chinchilla: Training Compute-Optimal Large Language Models

> **Hoffmann, Borgeaud, Mensch et al., DeepMind. 2022.** [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)

## Por que inclui este paper num módulo de otimização

Otimizadores e schedules decidem **como** treinar. Scaling laws decidem **quanto** treinar. Em 2022 a comunidade descobriu que estava otimizando sob um regime errado — modelos eram pequenos demais para o dataset ou, mais frequentemente, modelos eram grandes demais para o dataset. Chinchilla reescreveu essas regras de bolso, e toda a prática de otimização em 2023-2026 pressupõe o regime Chinchilla-like ou um derivado pós-Chinchilla.

## A descoberta central

Para um orçamento fixo de compute $C$ (em FLOPs), pergunte: qual a divisão ótima entre número de parâmetros $N$ e número de tokens de treino $D$ (assumindo $C \approx 6 N D$)? Kaplan et al. (2020) haviam sugerido que $N$ deveria crescer ~3× mais rápido que $D$. Chinchilla mostra o oposto:

> "For compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled."

Em números, ~**20 tokens por parâmetro** é o ponto compute-optimal. Traduzindo: um modelo de 70B parâmetros deveria ser treinado em ~1.4T tokens; um de 175B, em ~3.5T tokens. A surpresa é que **o GPT-3 (175B) havia sido treinado em apenas 300B tokens** — estava *subtreinado* por um fator de ~10×.

## As três metodologias

Os autores chegam à lei de escala por três caminhos independentes, o que aumenta drasticamente a confiança no resultado:

1. **IsoFLOP**: fixar diferentes orçamentos $C$ e treinar uma família de pares $(N, D)$ para cada; observar qual par minimiza a loss. Os mínimos formam uma curva que, em log-log, é uma reta — cuja inclinação dá a regra de scaling.
2. **IsoLoss**: fixar um alvo de loss e encontrar o $(N, D)$ mínimo-compute que atinge o alvo.
3. **Parametric form fit**: ajustar o modelo paramétrico $L(N, D) = E + A/N^\alpha + B/D^\beta$ aos dados empíricos de >400 modelos treinados entre 70M e 16B parâmetros.

Os três convergem à mesma conclusão: $\alpha \approx \beta$, portanto $N$ e $D$ devem escalar juntos.

## O teste: Chinchilla 70B vs Gopher 280B

Para validar empiricamente, DeepMind treinou **Chinchilla (70B params, 1.4T tokens)** usando o mesmo compute total de **Gopher (280B params, 300B tokens)**. Resultado:

- Chinchilla ganha em praticamente todo benchmark.
- +7 pontos em MMLU médio (67.5% vs 60%).
- Menor compute de fine-tuning e inferência (porque o modelo é 4× menor).

Essa é a demonstração definitiva de que "modelo maior" não era a pergunta certa — era "modelo mais treinado".

## Evolução pós-Chinchilla (2024-2026)

A lei de Chinchilla é compute-optimal para **treino apenas**. Se você pretende servir o modelo a muitos usuários (inference amortizado em milhões de tokens/dia), **vale a pena treinar um modelo menor por mais tempo** — o custo de inferência domina. Sardana et al. 2024 formalizaram esse "inference-aware scaling" e mostraram que o ponto ótimo pode ser **50-100 tokens/parâmetro** para uma vida útil típica de LLM em produção.

Exemplos em 2025:

- **LLaMA 3.1 70B**: treinado em 15T tokens = ~215 tokens/parâmetro.
- **Qwen3-0.6B**: 60 000 tokens/parâmetro — recorde de over-training para modelos pequenos.
- **Farseer (2024)** refina a predição Chinchilla com erro 4× menor em extrapolação para modelos grandes.

## Implicação para otimização

No regime Chinchilla / pós-Chinchilla (regime de **sub-treino**), o modelo processa cada token ~1-3 vezes em média. Isto muda tudo sobre otimização:

- Overfitting como no §5.5 do D2L **não ocorre**.
- Weight decay age via dinâmica, não via regularização de capacidade (ver Paper 2).
- Dropout não é útil (ver extensão de §5.6).
- Schedules longos com decay lento (WSD, D2Z) superam cosine-a-10% (ver Papers 8 e 10).
- Cross-validation clássica como descrita no §5.5 é impraticável (treinar é caro demais) — valida-se por loss curve e benchmarks.

## Limitações do paper original

- A lei $L(N, D)$ foi ajustada com dados até ~16B parâmetros; extrapolações para 1T+ são especulativas.
- O regime de dados é um snapshot de qualidade e diversidade particulares; duplicatas, contaminação e data mix afetam a constante (não o expoente).
- A lei assume "compute-optimal training" — para inference-optimal, reescreve-se.

## Status em 2026

"Chinchilla-optimal" virou parte do vocabulário básico. Qualquer paper sobre otimizador moderno reporta resultados ao longo da fronteira compute-optimal ou declara explicitamente se operando em regime diferente. Muon (Paper 5), Sophia (Paper 4), Schedule-Free (Paper 6) todos mencionam Chinchilla como baseline de referência de "quanto compute investido" — o eixo $x$ dos gráficos modernos de loss é frequentemente "Chinchilla-optimal units".
