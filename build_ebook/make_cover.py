"""Generate a clean, professional cover PNG using Pillow.

Design:
- 1600 x 2400 (EPUB recommended aspect ratio)
- Deep-teal to ink-black gradient background
- Minimalist typography in white
- Thin accent line
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "cover.png"

W, H = 1600, 2400


def make_gradient(w: int, h: int, top=(14, 36, 48), bottom=(3, 8, 14)) -> Image.Image:
    img = Image.new("RGB", (w, h), top)
    px = img.load()
    for y in range(h):
        t = y / (h - 1)
        r = int(top[0] * (1 - t) + bottom[0] * t)
        g = int(top[1] * (1 - t) + bottom[1] * t)
        b = int(top[2] * (1 - t) + bottom[2] * t)
        for x in range(w):
            px[x, y] = (r, g, b)
    return img


def find_font(preferred: list[str], size: int) -> ImageFont.FreeTypeFont:
    candidates = preferred + [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_text_centered(draw: ImageDraw.ImageDraw, text: str, font, y: int, fill=(255, 255, 255)):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) // 2, y), text, font=font, fill=fill)


def main() -> None:
    img = make_gradient(W, H)
    draw = ImageDraw.Draw(img)

    # Accent line near the top
    y_line = 320
    draw.line([(200, y_line), (W - 200, y_line)], fill=(220, 180, 110), width=3)

    # Top label
    label_font = find_font(["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"], 52)
    draw_text_centered(draw, "MÓDULO 2 · MSC AI · UC BOULDER", label_font, 220, fill=(220, 180, 110))

    # Main title (stacked)
    title_font = find_font([], 128)
    draw_text_centered(draw, "Otimização em", title_font, 620)
    draw_text_centered(draw, "Deep Learning", title_font, 770)

    # Subtitle
    sub_font = find_font([], 56)
    draw_text_centered(draw, "Base (Dive into Deep Learning)", sub_font, 1080)
    draw_text_centered(draw, "+ Atualizações 2023 → 2026", sub_font, 1160)

    # Central ornamental equation
    eq_font = find_font([], 44)
    eq = "θ_{t+1} = θ_t − η · \u2207L(θ_t)"
    draw_text_centered(draw, eq, eq_font, 1500, fill=(220, 220, 220))

    # Bottom topics list
    topics_font = find_font([], 38)
    topics = [
        "Weight Decay · Generalization · Dropout",
        "SGD · Momentum · Adam · LR Scheduling",
        "AdamW · Lion · Sophia · Muon · Schedule-Free",
        "Grokking · Double Descent · Scaling Laws",
    ]
    y = 1880
    for t in topics:
        draw_text_centered(draw, t, topics_font, y, fill=(190, 210, 220))
        y += 68

    # Bottom accent + date
    draw.line([(200, H - 260), (W - 200, H - 260)], fill=(220, 180, 110), width=2)
    date_font = find_font([], 40)
    draw_text_centered(draw, "Edição bilíngue · Abril de 2026", date_font, H - 200, fill=(220, 180, 110))

    img.save(OUT, "PNG")
    print(f"cover written → {OUT}")


if __name__ == "__main__":
    main()
