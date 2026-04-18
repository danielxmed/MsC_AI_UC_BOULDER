# Paper 5 — Muon: Momentum Orthogonalized (2024 + 2025)

> **Keller Jordan et al. 2024.** [Blog canônico](https://kellerjordan.github.io/posts/muon/) · [GitHub](https://github.com/KellerJordan/Muon)
>
> **Liu et al., Moonshot AI, 2025.** [arXiv:2502.16982](https://arxiv.org/pdf/2502.16982) — *Muon is Scalable for LLM Training.*

## Por que Muon é o sucessor natural do §12.6

O §12.6 do D2L trata momentum como aceleração: acumular a direção recente para amortecer oscilações. Muon olha momentum com **lentes geométricas**: o update acumulado é uma matriz, e matrizes têm **estrutura espectral**. Se a maior parte da "energia" do momentum está em poucas direções (rank baixo, alto número de condição), então o otimizador gasta compute aprendendo nessas direções dominantes enquanto as direções raras — que podem ser mais importantes para aprender features novos — ficam subexploradas. Muon ataca exatamente esse desequilíbrio.

## O algoritmo

Para cada parâmetro $W$ que é uma **matriz 2D de camada oculta** (todas as `nn.Linear` de transformer, exceto embeddings e head de saída):

1. **Momentum clássico (estilo Nesterov)**:
   $$M_t = \mu\, M_{t-1} + g_t,\quad G_t = g_t + \mu\, M_t$$
2. **Ortogonalização via Newton-Schulz**: aproximar a matriz ortogonal mais próxima de $G_t$ pela seguinte iteração, 5 vezes, com coeficientes fine-tuned $(a,b,c)=(3.4445, -4.7750, 2.0315)$:
   $$X \leftarrow a\,X + b\,(X X^\top) X + c\,(X X^\top)^2 X$$
3. **Update**:
   $$W \leftarrow W - \eta\, \operatorname{NS}(G_t)$$

Matematicamente: se $G = U \Sigma V^\top$ é a SVD, a iteração NS converge para $U V^\top$ — a matriz ortogonal mais próxima de $G$ na norma de Frobenius. Essa operação **amplifica as direções singulares raras** (pequeno $\sigma$) ao mesmo nível das dominantes, efetivamente "rebalanceando" o update.

## Estabilidade em bf16 — vantagem decisiva

Newton-Schulz executa **estavelmente em bfloat16**. Iteração de Newton clássica (passo $X_{k+1} = X_k(2I - A X_k)$) exige float32 por sensibilidade numérica. Essa diferença é o que torna Muon viável para LLMs: todo o restante do pipeline moderno (forward, backward, attention) roda em bf16 para reduzir memória e aumentar throughput; um otimizador que force upgrade para fp32 quebra esse compromisso.

## Aplicação seletiva

Muon **não é aplicado a tudo**:

- **Embeddings e head de output**: ficam em AdamW. A teoria de "modular norm" explica o caso do embedding; o output é empírico.
- **Biases e ganhos de normalização**: ficam em AdamW (são 1D, Muon não se aplica).
- **Convoluções**: tratadas como `Linear` após achatar as últimas 3 dimensões — entram em Muon.
- **Projeções Q/K/V de attention**: tratadas como três matrizes separadas, cada uma com sua ortogonalização independente.

Resultado: um treino típico é **híbrido**: a maioria dos parâmetros em Muon, alguns poucos (embedding, head, biases, norm) em AdamW.

## Evidência empírica — escala pequena e média

- **NanoGPT speedrun** (meta: atingir val loss 3.28 em FineWeb): Muon reduziu o recorde em **1.35×** em outubro 2024; 12+ recordes sucessivos desde então por 7 pesquisadores independentes — o padrão mais rigoroso de evidência empírica disponível em otimização.
- **CIFAR-10 94%**: de 3.3 → 2.6 A100-segundos.
- **1.5B params atingindo performance de GPT-2 XL em HellaSwag** em 10h com 8×H100, vs 13.3h com AdamW — ~25% redução wall-clock.

## Evidência empírica — escala de fronteira

Em 2025, Moonshot AI publicou *Muon is Scalable for LLM Training*:

- **Moonlight** — Mixture-of-Experts com 3B e 16B parâmetros, 5.7T tokens: **~2× eficiência vs AdamW compute-optimal**. Movimenta a fronteira de Pareto.
- **Kimi K2** — modelo trilionário da Moonshot, treinado **inteiramente com Muon**. É a maior validação em produção.
- Duas adições ao Muon original foram cruciais para escala: (1) adicionar weight decay, (2) escalar updates por-parâmetro de forma calibrada. Sem estas, Muon puro em trilhão divergia.

## Custo computacional

O overhead de Newton-Schulz: $5m$ FLOPs por parâmetro (com $m$ sendo a menor dimensão da matriz), comparado a $6mB$ no forward-backward (com $B$ = tokens por batch). Em NanoGPT ($m=768, B=524288$): ~0.7%. Em Llama 405B ($m=16384, B=16M$): ~0.5%. Ou seja, overhead sub-1% em cenários realistas.

## Status em 2026

Muon é o otimizador **mais promissor** entre os novos — superou AdamW em benchmarks sucessivos, tem deployment trilionário validado, e tem o manual de uso da comunidade open-source razoavelmente maduro. Ainda não é default universal (AdamW domina por maturidade), mas é a aposta mais forte para pretraining de frontier em 2026-2027.

## Implicação pedagógica

Muon mostra que momentum não é apenas uma heurística de aceleração — é um **tensor estruturado** cuja geometria importa. Ideias análogas de ortogonalização de estado apareceram em Shampoo/SOAP (próximo capítulo conceitual) mas Muon é o primeiro a combinar simplicidade, estabilidade em bf16 e validação em escala trilionária.
