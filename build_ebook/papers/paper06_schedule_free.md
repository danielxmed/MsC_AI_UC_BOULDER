# Paper 6 — Schedule-Free: The Road Less Scheduled

> **Defazio, Yaida, Mishchenko, Khaled, Cutkosky. NeurIPS 2024.** [arXiv:2405.15682](https://arxiv.org/abs/2405.15682) · [GitHub](https://github.com/facebookresearch/schedule_free)

## A premissa radical

Todo schedule do §12.11 do livro — cosine, polinomial, step decay — compartilha uma exigência implícita: **você precisa saber, de antemão, quantos passos vai treinar**. A LR decai ao longo de $T$ conhecido. Schedule-Free questiona essa premissa e mostra que, matematicamente, podemos **eliminar o schedule** preservando (ou melhorando) a qualidade final.

## Intuição — Polyak–Ruppert + Primal averaging

Em otimização convexa, o teorema de Polyak-Ruppert diz que se você rodar SGD com LR constante e reportar a **média dos iterados**, a variância do estimador cai como $O(1/\sqrt{t})$ sem precisar decaimento de LR. A catch é que em não-convexo (nosso caso), isso não funciona diretamente.

Schedule-Free combina dois tipos de averaging — Polyak-Ruppert (média de iterados) e primal averaging (média do ponto de avaliação) — de forma a obter as garantias de convergência equivalentes ao melhor schedule de LR, *sem usar schedule*. A elegância é notável: com LR constante o método tem convergência ótima, e não precisa saber $T$ antecipadamente.

## As três sequências

A mecânica em pseudocódigo:

```
y_t = (1 - β) · z_t + β · x_t     # ponto de avaliação do gradiente
z_{t+1} = z_t - η · ∇L(y_t)       # update estilo AdamW (sem schedule)
x_{t+1} = (1 - c_t) · x_t + c_t · z_{t+1}   # média acumulada
```

- $z$ é a sequência de "passos" (onde o update de gradiente é aplicado).
- $x$ é a sequência de "pontos reportados" (a média de Polyak-Ruppert).
- $y$ é a sequência de "pontos de avaliação" do gradiente (interpolação entre $z$ e $x$).

O coeficiente $c_t = 1/(t+1)$ faz $x$ ser a média aritmética simples; $\beta$ controla a força da interpolação primal ($\beta=0$ degenera para Polyak-Ruppert puro).

## Garantias e resultados

Os autores provam convergência ótima (matching bounds) para convexos, e empiricamente:

- **Venceu a pista Self-Tuning do MLCommons AlgoPerf 2024**, que avalia otimizadores sem hiperparâmetros ajustáveis. Essa é a única competição em otimização com protocolo rigoroso.
- Em LLM pretraining, **iguala AdamW com cosine decay** em compute matched; em cenários onde o usuário não sabe $T$ com precisão, supera.
- LR ótima para Schedule-Free é tipicamente **1×-10× maior** que a pico de uma LR-scheduled equivalente — faz sentido, já que não há decaimento.

## Por que isso importa na prática

1. **Treinos indefinidos**: se você quer parar "quando der" (ex: infraestrutura compartilhada, budget variável), Schedule-Free entrega qualidade sem se comprometer com $T$.
2. **Evita erro catastrófico**: se você chutou $T$ errado com cosine e treinou além, o LR residual ($0.1 \eta_{max}$) continua caindo mas você pode ficar preso em região ruim. Schedule-Free não tem essa armadilha.
3. **Simplifica experimentação**: não precisa re-tunar schedule quando escala o modelo.

## Ressalva importante: BatchNorm/LayerNorm

O averaging tem uma interação sutil com camadas de normalização que rastreiam estatísticas online (BN/LN). Os pesos dos layers de normalização não devem ser averaged da mesma forma — a implementação de referência trata isso, mas quem for integrar em código custom precisa estar atento. Se você esquecer, o modelo treinado parece OK mas tem desempenho degradado em avaliação.

## Implementação

```bash
pip install schedulefree
```

```python
from schedulefree import AdamWScheduleFree

opt = AdamWScheduleFree(model.parameters(), lr=3e-4, warmup_steps=100, weight_decay=0.1)

# Training
opt.train()  # põe em modo treino
for batch in train_loader:
    loss = compute_loss(model, batch)
    loss.backward()
    opt.step()
    opt.zero_grad()

# Avaliação
opt.eval()  # usa a média x para forward — crítico
with torch.no_grad():
    eval_metrics(model, val_loader)
opt.train()  # volta para treino
```

A distinção `opt.train()` vs `opt.eval()` é essencial: durante avaliação, os pesos usados são $x$ (média), durante treino são $y$ (ponto de avaliação).

## Status em 2026

Schedule-Free é o único otimizador moderno com vitória em competição com protocolo blind (MLCommons). Está ganhando adoção em pipelines que querem reduzir hiperparâmetros, mas não deslocou cosine/WSD em pretraining de frontier — mais pela inércia institucional do que por inferioridade técnica. Um excelente candidato para experimentos de research onde o budget não é conhecido ex-ante.
