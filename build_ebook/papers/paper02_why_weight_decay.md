# Paper 2 — Why Do We Need Weight Decay in Modern Deep Learning?

> **Andriushchenko, D'Angelo, Varre, Flammarion. NeurIPS 2024.** [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/29496c942ed6e08ecc469f4521ebfff0-Paper-Conference.pdf) · [código](https://github.com/tml-epfl/why-weight-decay)

## Tese que contraria o livro

O §3.7 do D2L apresenta weight decay como **regularização no sentido clássico**: o termo $\lambda\|\theta\|^2/2$ reduz a capacidade efetiva do modelo e, portanto, o overfitting. Este paper demonstra que em aprendizado profundo moderno essa leitura é **essencialmente incorreta**. A sentença que resume a tese:

> *"Weight decay is never useful as an explicit regularizer but instead changes the training dynamics in a desirable way."*

Isto é, o benefício de aplicar weight decay não vem de "segurar a norma dos pesos", mas de como ele altera a trajetória do otimizador ao longo do espaço de parâmetros. O paper identifica **dois mecanismos distintos** conforme o regime de treino.

## Mecanismo 1: redes de visão, múltiplas épocas

No regime que o livro tem em mente — ResNets/ViTs em ImageNet com múltiplas épocas e possibilidade real de overfit — o weight decay age **amplificando o ruído implícito do SGD**. A análise faz a seguinte conexão:

1. SGD injeta ruído estruturado no subespaço dos gradientes por exemplo.
2. Esse ruído, combinado com a norma do Jacobiano, gera um efeito de regularização implícita que controla a sharpness do mínimo final.
3. Weight decay **mantém** esse efeito ativo ao impedir que a norma dos pesos cresça indefinidamente (o que atenuaria o ruído).

Ou seja, o benefício não é a penalidade $\lambda\|\theta\|^2/2$ em si, mas sim que o decay mantém o SGD operando no regime onde o implicit bias é efetivo. Isto explica por que **weight decay combina particularmente bem com batch sizes pequenos e LR grandes** — todos os ingredientes do "SGD noise regime".

## Mecanismo 2: LLMs, uma única época

Para grandes modelos de linguagem em pretraining com ~1 época sobre os dados, o livro não se aplica: **não há overfitting no sentido clássico** (o modelo nunca reencontra o mesmo token). Aqui o papel do weight decay é ainda mais distante da regularização:

- Ele equilibra o *bias-variance tradeoff* do próprio estimador estocástico do gradiente;
- Reduz o **training loss** (não só o test loss), o que invalida completamente a leitura "sacrifica-se treino por generalização";
- **Previne divergências repentinas em mixed precision bfloat16** — as chamadas *sudden loss spikes* que são o fantasma dos treinos em larga escala.

Este último ponto é decisivo: em bf16, o range dinâmico é amplo mas a mantissa curta (7 bits) introduz ruído numérico. Weight decay estabiliza porque mantém a norma dos pesos em uma faixa onde os produtos $W \cdot x$ permanecem em ordem de grandeza consistente, evitando overflow/underflow intermitentes.

## Evidências experimentais

Os autores validam cada mecanismo separadamente:

- **Em visão** — ResNets em CIFAR-10/100 e ImageNet — mostram que remover weight decay degrada tanto test loss quanto sharpness (medida via maior autovalor do Hessiano); restabelecer o decay reverte ambos.
- **Em LLMs** — modelos da família GPT-2 e maiores — mostram que a presença de weight decay reduz a variância trecho-a-trecho da loss curve e reduz a frequência de loss spikes em bf16. O training loss final é menor *com* weight decay.

## Implicações práticas

1. **Mantenha weight decay ligado** em LLM pretraining, mesmo sabendo que não há overfit — ele é parte da tríade anti-NaN (junto com gradient clipping e warmup).
2. **$\lambda \in [0.01, 0.1]$** é a faixa robusta; $\lambda=0.1$ é típico em pretraining, $0.01$ em fine-tuning.
3. **Exclua biases e parâmetros de normalização** (LayerNorm/RMSNorm γ) do decay — eles não se beneficiam do mecanismo de estabilização dinâmica.
4. **Não espere melhora em test loss** via aumento de $\lambda$ em LLMs — a curva é aproximadamente plana sobre uma faixa ampla de $\lambda$; escolha pela estabilidade, não pela generalização.

## Conexão com o restante do material

Este paper explica **por que a intuição clássica do §3.7 envelheceu** sem invalidar a prática: continuamos ligando weight decay, mas por razão diferente da que o livro ensina. Combinando com o paper AdamW (anterior), temos o quadro moderno: use AdamW com $\lambda=0.1$, bf16, warmup curto, clip de 1.0 — não porque cada peça regulariza capacidade, mas porque juntas geram uma dinâmica estável em escala.
