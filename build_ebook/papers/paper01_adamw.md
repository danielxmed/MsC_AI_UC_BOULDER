# Paper 1 — AdamW: Decoupled Weight Decay Regularization

> **Loshchilov & Hutter, ICLR 2019.** [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)

## Por que este paper importa para o módulo

O §12.10 do *Dive into Deep Learning* apresenta Adam como apresentado originalmente em 2015 — com `weight_decay` somado ao gradiente como uma penalidade $\ell_2$. Este paper mostra, formal e empiricamente, que **essa prática estava errada** para otimizadores adaptativos. A correção — AdamW — é hoje o otimizador-padrão em praticamente todo pretraining de transformer. Compreender *por que* AdamW difere de Adam + L² é pré-requisito para ler qualquer código de LLM moderno.

## A descoberta central

> "L² regularization and weight decay regularization are equivalent for standard stochastic gradient descent (when rescaled by the learning rate), but as we demonstrate this is not the case for adaptive gradient algorithms, such as Adam."

Em SGD, somar $\lambda\theta$ ao gradiente produz exatamente o mesmo passo que $\theta \leftarrow (1-\eta\lambda)\theta - \eta g$. A equivalência se quebra em Adam porque o pré-condicionador adaptativo $1/(\sqrt{\hat v_t} + \epsilon)$ divide *também* a contribuição de $\lambda\theta$ — fazendo com que pesos grandes em direções de baixa variância do gradiente (onde $\hat v_t$ é pequeno) sejam *relativamente mais* decaídos do que em direções de alta variância. Ou seja: "weight decay" via L² no Adam fica **acoplado** aos segundos momentos, o que não é o comportamento desejado.

## A correção: AdamW

A solução é aplicar o decaimento **fora** do passo adaptativo:

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat m_t &= m_t / (1 - \beta_1^t), \quad \hat v_t = v_t / (1 - \beta_2^t) \\
\theta_t &= \theta_{t-1} - \eta_t\Big(\underbrace{\hat m_t / (\sqrt{\hat v_t} + \epsilon)}_{\text{passo adaptativo}} + \underbrace{\lambda\, \theta_{t-1}}_{\text{decay desacoplado}}\Big)
\end{aligned}
$$

A implementação é tão simples quanto trocar `torch.optim.Adam` por `torch.optim.AdamW`. Essa troca de uma linha é responsável por boa parte das melhorias empíricas atribuídas a pequenas variantes arquiteturais em papers de 2019-2021 que não as isolavam cuidadosamente.

## Evidência empírica

Nos experimentos do paper, AdamW reduz substancialmente a lacuna de *top-1 accuracy* entre Adam e SGD-com-momentum em ImageNet e CIFAR, permitindo que Adam se torne competitivo em tarefas de visão — exatamente o contexto em que antes se dizia "Adam converge rápido mas generaliza pior". Outro resultado importante: **o $\lambda$ ótimo fica aproximadamente independente da taxa de aprendizado**, simplificando drasticamente a sintonia conjunta de hiperparâmetros.

## Adoção na prática

AdamW é hoje o default de PyTorch (`torch.optim.AdamW`), Hugging Face `transformers`, DeepSpeed, PyTorch Lightning, JAX/Optax (`optax.adamw`). Qualquer repositório sério de LLM — LLaMA, Mistral, Qwen, DeepSeek, GPT-NeoX, Pythia — usa AdamW ou uma variante de decay desacoplado. Os defaults típicos são $\beta_1=0.9, \beta_2=0.95, \epsilon=10^{-8}$ (1e-7 em bf16), $\lambda=0.1$ com warmup de 1-10% dos passos e LR-peak na casa de $3 \times 10^{-4}$ para modelos de 1-70B.

## Limitações do paper

O escopo empírico é dominado por visão (CIFAR-10/100, ImageNet) em CNNs. A validação em NLP/transformers veio só com a adoção pela comunidade nos 2-3 anos seguintes. O artigo também não fornece análise teórica de convergência da nova dinâmica — ela existe em trabalhos posteriores, mas continua menos madura que a teoria de Adam clássico.

## Conexão com o capítulo seguinte

O paper de Andriushchenko et al. (2024, próximo capítulo) argumenta que **mesmo com AdamW, a interpretação "weight decay regulariza capacidade" está errada** — o papel real é dinâmico. Essa camada teórica é a atualização 2024 sobre o próprio legado do AdamW.
