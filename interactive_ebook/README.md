# Otimização em Deep Learning — versão interativa

Versão web interativa do ebook `deep_learning_modulo_2_ebook.epub`, com busca
client-side, dark mode, navegação por swipe (mobile), marcador de progresso,
lightbox de imagens e atalhos de teclado.

O conteúdo (texto, equações MathML e imagens) é extraído **exatamente** do
EPUB canônico — nenhuma palavra é perdida ou reescrita.

## Live

Após o primeiro push para `main`, o GitHub Pages publica automaticamente em:

    https://<user>.github.io/<repo>/

Para descobrir a URL exata:

```bash
gh repo view --json nameWithOwner -q .nameWithOwner
```

Setup único no GitHub: **Settings → Pages → Source: GitHub Actions**.

## Como ler no iPhone / iPad

1. Abra a URL acima no **Safari** (não Chrome — o *Add to Home Screen* do
   Chrome iOS é limitado).
2. Toque em **Compartilhar** → **Adicionar à Tela de Início**.
3. O ícone (derivado da capa) aparece na home. Tocar abre em modo
   *standalone* (sem barra de URL) e, após a primeira abertura online, o
   ebook funciona **offline para sempre** — avião, metrô, sem sinal.

## Como ler offline no desktop

```bash
git clone <repo>
cd <repo>/interactive_ebook
python3 -m http.server 8000
# abra http://localhost:8000
```

Abrir `index.html` diretamente com `file://` também funciona, porém o
service worker só registra em `http(s)://`.

## Features

- **Busca**: `/` foca a caixa de busca (filtra TOC + destaca ocorrências).
- **Dark mode**: tecla `d` ou o botão no topbar. Persistente via
  `localStorage`.
- **Navegação**:
  - Desktop: `j`/`k` scroll, `←`/`→` capítulo anterior/próximo, `t` TOC,
    `Esc` fecha busca/lightbox, `?` mostra legenda.
  - Mobile: swipe horizontal (threshold 80px) para capítulo anterior/próximo.
- **Lightbox**: clique/tap em qualquer imagem amplia em overlay. Fecha por
  `Esc`, tap fora ou ✕.
- **Código**: botão “copiar” em cada `<pre>`.
- **Marcador**: capítulos lidos ganham ✓ no TOC (via `localStorage`).
- **Filtros de status**: botões 🟢 / 🟡 / 🔴 no topbar filtram conteúdo
  por status (atual / parcial / legado).

## Como atualizar

O conteúdo é regenerado a partir do EPUB via:

```bash
python3 build_ebook/build_interactive.py
```

O workflow `.github/workflows/pages.yml` redeploy automaticamente em cada
push para `main` que toque `interactive_ebook/**`.
