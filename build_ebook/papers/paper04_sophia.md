# Paper 4 — Sophia: Second-order Optimizer for Language Model Pre-training

> **Liu, Tian, Chen, Sanborn, Ma, Goodman. ICLR 2024.** [arXiv:2305.14342](https://arxiv.org/abs/2305.14342)

## O que o paper tenta resolver

Métodos de segunda ordem (Newton, quase-Newton, K-FAC, Shampoo) têm teoria superior ao SGD/Adam, mas na prática são pesados demais para LLM pretraining — o custo de calcular ou aproximar o Hessiano a cada passo destrói qualquer ganho de convergência. Sophia busca o ponto de equilíbrio: **usar uma aproximação diagonal do Hessiano, atualizada infrequentemente (a cada ~10 passos), com clipping elemento-a-elemento** para controlar updates extremos. O resultado é overhead quase nulo com ganhos reais.

## A atualização de Sophia

A mecânica do Sophia combina três ideias. Primeiro, uma média móvel dos gradientes como no Adam:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t$$

Segundo, uma estimativa diagonal do Hessiano $\hat H_t$ — os autores usam **Gauss-Newton-Bartlett** (GNB), barato para modelos com perda cross-entropy — calculada apenas a cada $k$ passos (tipicamente $k=10$). Entre atualizações, $\hat H_t$ é reutilizado. Uma média móvel estabiliza a estimativa:

$$h_t = \beta_2 h_{t-1} + (1-\beta_2)\, \hat H_t^{\text{diag}}$$

Terceiro — e é a peça decisiva — um **clipping elemento a elemento** do pré-condicionamento:

$$
\theta_t = \theta_{t-1} - \eta\,\operatorname{clip}\!\Big(m_t \big/ \max(\gamma\,h_t,\, \epsilon),\ -\rho,\ +\rho\Big)
$$

O clipping garante que nenhum componente do update exceda magnitude $\rho$ (tipicamente $\rho=1$). Isto "tames the negative impact of non-convexity and rapid change of Hessian along the trajectory" — ou seja, controla o caso em que $h_t$ é tão pequeno que o update explode em uma direção. Na prática, o pré-condicionamento é aplicado *suavemente*, e o clipping age como safety net.

## Por que funciona

A intuição chave: em paisagens de loss de LLMs, **curvaturas diferentes em diferentes direções** são a norma. Adam adapta por-parâmetro via $v_t$, mas $v_t$ é a variância do gradiente, não a curvatura da loss — elas coincidem só para objetivos lineares. Sophia usa a curvatura estimada diretamente, atacando o mesmo problema com ferramenta mais apropriada.

## Resultados empíricos

Nos experimentos do paper (GPT-2 classes, 125M a 1.5B parâmetros, treinando no OpenWebText):

- **2× speedup vs AdamW** em passos, compute total e wall-clock — a Sophia atinge a mesma perplexity com 50% menos compute.
- O ganho **cresce com o tamanho do modelo**, sugerindo que Sophia é mais eficaz em modelos maiores — tendência animadora para pretraining de frontier.
- Overhead por passo é aproximadamente constante, pois o cálculo GNB é feito só a cada 10 passos e o resto do pipeline é comparável ao AdamW.

## Custo e implementação

O GNB Hessian estimator requer um backward adicional a cada 10 passos com uma amostragem de pseudo-labels sampleados do modelo — barato em termos amortizados (~1-3% overhead médio). O estado do otimizador fica: $m$ + $h$ = 2× os pesos, mesmo tamanho que AdamW. Não há ganho de memória, só de velocidade.

## Limitações

- **Sensibilidade à implementação** — $\rho$, $\gamma$, $k$ e $\beta_2$ formam um espaço de tuning razoavelmente delicado.
- **Análise teórica simplificada** — a prova de convergência dos autores assume cenário ideal; transferência para o regime real de LLM é mais empírica.
- **Adoção em produção é menor que a comunidade prevê**. Apesar dos resultados, AdamW e Muon capturam a maior parte do mindshare em 2026, porque a diferença prática em escala de trilhão de parâmetros ainda não foi demonstrada publicamente.

## Status em 2026

Sophia ocupa um nicho interessante: é o otimizador de segunda ordem mais próximo de "drop-in replacement" do AdamW que temos hoje, mas o nível de confiança depositado no Muon (que foi usado para treinar o Kimi K2 trilionário) é hoje maior. Vale conhecer como referência — e como ponto de comparação para o próximo salto de otimizadores de segunda ordem práticos.
