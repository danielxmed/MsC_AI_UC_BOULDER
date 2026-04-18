# Paper 3 — Lion: Symbolic Discovery of Optimization Algorithms

> **Chen, Liang, Huang, Real, Wang, Pham, Dong, Luong, Hsieh, Lu, Le. NeurIPS 2023.** [arXiv:2302.06675](https://arxiv.org/abs/2302.06675)

## A história incomum desta descoberta

Lion (*EvoLved Sign Momentum*) não foi desenhado manualmente por um pesquisador. Ele foi **descoberto por busca simbólica automatizada** no Google Brain: um sistema exploratório evoluiu programas candidatos de otimizador num espaço vasto de expressões, com regras de seleção e simplificação para reduzir overfit entre tarefas-proxy e tarefas reais. O algoritmo mais consistente que emergiu é **tão simples** que chocou os próprios autores.

## A atualização em duas linhas

O Lion só mantém um único estado auxiliar por parâmetro — o momentum $m$ — e produz updates de magnitude uniforme via `sign`:

$$
\begin{aligned}
u_t &= \operatorname{sign}\!\big(\beta_1 m_{t-1} + (1 - \beta_1)\, g_t\big) \\
m_t &= \beta_2 m_{t-1} + (1 - \beta_2)\, g_t \\
\theta_t &= \theta_{t-1} - \eta_t\,\big(u_t + \lambda\, \theta_{t-1}\big)
\end{aligned}
$$

Compare com AdamW: Lion descarta a estimativa $v_t$ do segundo momento (metade da memória) e a operação não-linear de divisão por $\sqrt{v_t}$. A perda de adaptação por-parâmetro é compensada pela operação `sign`, que normaliza o tamanho de cada componente do update.

Defaults recomendados: $\beta_1=0.9, \beta_2=0.99$ (note que $\beta_2$ aqui **não** é a do Adam), $\lambda=0.1 \times$ o valor que se usaria em AdamW, e $\eta$ cerca de **3–10× menor** que em AdamW equivalente.

## Por que Lion funciona

A interpretação mais convincente é que `sign(m)` **uniformiza o tamanho do passo por coordenada**, o que faz cada parâmetro caminhar na mesma velocidade — uma forma agressiva de pré-condicionamento. Em paisagens mal condicionadas (típicas de redes grandes), isso explora bem direções raras. A desvantagem é que em direções onde o gradiente varia com sinal oposto em batches consecutivos, o momentum atrasa a estabilização — daí a necessidade de $\beta_1=0.9$ (mais conservador que o típico) para garantir que o sinal seja o do momentum, não do gradiente corrente.

## Resultados empíricos de destaque

No paper e reproduções independentes:

- **Visão (ViT em ImageNet)**: até +2% de top-1 accuracy vs AdamW; economia de até 2× no compute de pretraining em JFT.
- **Difusão (Imagen, Stable Diffusion-class)**: FID **2.3× melhor** e redução do compute de treino de até 2.3×.
- **Contrastivo visão-linguagem (CLIP-class)**: 88.3% zero-shot e 91.1% fine-tuning em ImageNet (+2% e +0.1% vs melhores até então).
- **Language modeling autoregressivo**: desempenho similar ou ligeiramente melhor que Adam, mais sensível a hiperparâmetros.
- Notavelmente, Lion foi **deployado em produção no Google search ads CTR model**, o que é um sinal forte de que não é só um resultado benchmark.

## Economia de memória

Para um modelo de 70B parâmetros, AdamW guarda $m$ e $v$ (= 2× os pesos = ~280 GB em bf16). Lion guarda só $m$ (= ~140 GB), liberando metade da memória do otimizador. Isto não é trivial — em muitos setups, o estado do otimizador é o segundo maior consumo de VRAM depois das próprias ativações.

## Limitações reconhecidas

Os próprios autores e vários follow-ups observam que:

1. **Sensibilidade a LR e weight decay** — faixas estreitas; tunar é mais crítico que em AdamW.
2. **Menor estabilidade em escala muito grande** para LLM pretraining puro — benchmarks para modelos >7B mostram resultados mistos.
3. **Benefício maior quando o batch é grande** — em regime de batch pequeno, a ausência de $v_t$ deixa a variância do update mais alta.

## Status em 2026

Lion é o **default padrão de facto em pretraining de modelos de difusão** (Stable Diffusion 3, Flux). Em LLMs competitivos com AdamW em várias faixas, mas raramente vence por margem decisiva — daí AdamW continuar sendo a escolha segura em pretraining de frontier models. A lição metodológica do paper (busca simbólica de otimizadores) é possivelmente mais influente que o algoritmo final.
