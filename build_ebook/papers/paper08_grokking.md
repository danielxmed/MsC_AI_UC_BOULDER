# Paper 8 — Explaining Grokking through Circuit Efficiency

> **Varma, Shah, Kenton, Kramár, Kumar. 2023.** [arXiv:2309.02390](https://arxiv.org/pdf/2309.02390)

## O fenômeno do grokking

Em 2022, Power et al. observaram um padrão bizarro ao treinar transformers em tarefas algorítmicas (ex: adição modular): o modelo atinge **100% de training accuracy em poucos passos**, enquanto validation accuracy fica em ~chute por $10^4$ a $10^6$ passos, e então — de repente — **sobe abruptamente para 100%**. É a generalização tardia, ou *grokking*.

Este paper de Varma, Shah, Kenton et al. (2023) apresenta a explicação que, em 2025, é consenso na literatura: grokking é a **manifestação visível de uma competição entre dois circuitos**, onde o weight decay inclina a balança lentamente em favor do mais eficiente.

## A hipótese dos dois circuitos

Em uma tarefa suficientemente estruturada (como adição modular $a + b \mod p$), há pelo menos duas maneiras de chegar a 100% em treino:

1. **Circuito de memorização**: o modelo aprende uma tabela hash dos pares $(a, b) \to c$ presentes no training set. Tamanho de parâmetros usado é proporcional ao tamanho do training set.
2. **Circuito generalizante**: o modelo aprende a *estrutura algorítmica* — por exemplo, via representações de Fourier discreta sobre $\mathbb{Z}/p\mathbb{Z}$. Tamanho de parâmetros usado é aproximadamente constante (o custo fixo do algoritmo), independente do training set.

Ambos atingem 100% em treino. Mas em **validation** só o segundo funciona, porque a memorização não se estende a $(a,b)$ fora do training set.

## A eficiência decide

O insight: dado que ambos existem em peso igual no início, o que favorece a emergência do circuito generalizante? **Weight decay**. Considere a "eficiência" de um circuito — quanto logit ele produz por unidade de norma de pesos:

- Circuito de memorização: o custo em norma cresce linearmente com $|D_{train}|$.
- Circuito generalizante: o custo em norma é aproximadamente constante.

Conforme training set aumenta (ou equivalentemente, conforme o treino avança e mais tokens são vistos), a razão de eficiência muda. Em algum ponto, o circuito generalizante domina. Weight decay então age como pressão constante que **recompensa o vencedor de eficiência**, levando à transição abrupta.

## As previsões novas e validadas

O paper não só explica o grokking — ele faz predições novas que só sua teoria prediz:

1. **Ungrokking**: se você reduz o training set *depois* de grokar, o modelo desgrokará — reverte para o circuito de memorização. Isso é observado experimentalmente.
2. **Semi-grokking**: se o training set é pequeno o suficiente, nenhum circuito domina claramente; o modelo estabiliza em desempenho **parcial** em validation — não 100% nem 0%. Observado.
3. **Influência de $\lambda$**: aumentar weight decay acelera o grokking; reduzir retarda ou impede. Observado.
4. **Influência do size do modelo**: modelos muito pequenos nunca grokam (não cabem o circuito generalizante); modelos muito grandes grokam mais lento (memorização é barata). Curva em U. Observado.

## Validação empírica

As tarefas de teste são controladas e analisáveis:

- Adição modular $a + b \mod p$ em vários $p$.
- Composição de permutações em $S_5, S_7$.
- Pequenos grafos de tarefas algébricas.

Em todas, os autores demonstram o mecanismo isolando componentes (decomposição espectral dos pesos após grokar, intervenção cirúrgica em subcircuitos) e replicam os 4 efeitos preditos.

## Implicação para o livro

O §5.5 do D2L ensina que se training e validation loss divergem, o modelo não está generalizando. Grokking mostra que isso está **certo localmente mas errado globalmente** — a divergência inicial é transitória e pode mascarar generalização emergente. Concretamente:

1. **Cross-validation com paciência curta** pode classificar como "fracasso" um modelo que estava a um número grande de passos de grokar.
2. **Weight decay insuficiente** impede o grokking — o circuito memorizador fica competitivo demais.
3. O fenômeno é específico a tarefas estruturadas; em dados naturais ruidosos o grokking é menos abrupto, se manifestando como *slingshot effects* e *emergent capabilities* — observados em LLMs ao longo do treino.

## Conexão com lazy vs rich

Uma leitura alternativa (Clare Lyle 2025) é que grokking é a transição do regime **lazy** (NTK, features fixas) para o regime **rich** (features são atualizadas). Na fase lazy, o modelo memoriza; na fase rich, aprende representações. Weight decay e compute-suficiente empurram o modelo do primeiro para o segundo. As duas explicações são compatíveis — "eficiência de circuito" é *o que* muda; "lazy → rich" é *como* muda em termos de dinâmica dos pesos.

## Status em 2026

Grokking saiu do nicho e virou ferramenta conceitual para entender emergência em LLMs. A visualização interativa do Google PAIR (*Do ML Models Memorize or Generalize?*) é excelente para construir intuição. Para o módulo: grokking explica por que **weight decay positivo é essencial** mesmo no regime de "sem overfit" dos LLMs — ele é o que permite que circuitos mais eficientes emergem com treino suficiente.
