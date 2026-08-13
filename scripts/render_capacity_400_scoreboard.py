#!/usr/bin/env python3
"""Render the capacity-400 scoreboard from the leaderboard JSON.

The scoreboard used to be a hand-authored SVG, so every re-evaluation meant
editing bar widths by hand and the picture could silently disagree with the
numbers. This reads `perfect_cost_leaderboard_capacity_400.json` and emits both
the SVG and the PNG, so the chart is always whatever the evaluations actually
say.

    python scripts/render_capacity_400_scoreboard.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "artifacts/live_y_capacity_400/evaluations/perfect_cost_leaderboard_capacity_400.json"
SVG_OUT = ROOT / "docs/assets/live-y-capacity-400-benchmark.svg"
PNG_OUT = ROOT / "docs/assets/live-y-capacity-400-benchmark-v3.png"

W = 1120
TRACK_X0, TRACK_X1 = 142, 1040
TRACK_W = TRACK_X1 - TRACK_X0          # 100 score points span the full track
TITLE_Y = 56
RULE_Y = 116
HDR_Y = 130
ROW_Y0, ROW_STEP = 158, 56
CARD_PAD = 41                          # breathing room under the last bar


def canvas_height(n_rows: int) -> int:
    """Height follows the row count so the card is never mostly empty."""
    return ROW_Y0 + (n_rows - 1) * ROW_STEP + 41 + CARD_PAD + 24

BG = "#F2F5F4"; CARD = "#FCFDFC"; EDGE = "#D8E2E2"
INK = "#172B35"; MUTED = "#52666C"; FAINT = "#829196"
TEAL_DARK = "#0B5D6B"; TEAL_LIGHT = "#2BA29A"
OURS_DARK = "#8A4B1F"; OURS_LIGHT = "#E5A64E"
TRACK_BG = "#E6EEED"
BADGE = ["#E5A64E", "#C7D4D5", "#D38A62"]
BADGE_REST = "#E2EBEA"

FONTS = {
    "mono": "/System/Library/Fonts/Menlo.ttc",
    "serif": "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "sans": "/System/Library/Fonts/Supplemental/Arial.ttf",
}
# The repository's own adapter, flagged so a reader can tell it from an API model.
OURS = "Qwen3.5-4B GRPO"
BASE = "Qwen3.5-4B (untrained)"


def font(kind: str, size: int, scale: int, bold: bool = False):
    path = FONTS[kind]
    if kind == "sans" and bold:
        path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    try:
        return ImageFont.truetype(path, size * scale)
    except OSError:
        return ImageFont.truetype(FONTS["sans"], size * scale)


def hgrad(draw: ImageDraw.ImageDraw, box, c0: str, c1: str, scale: int) -> None:
    """Horizontal gradient inside a rounded bar."""
    x0, y0, x1, y1 = box
    if x1 <= x0:
        return
    r0 = tuple(int(c0[i : i + 2], 16) for i in (1, 3, 5))
    r1 = tuple(int(c1[i : i + 2], 16) for i in (1, 3, 5))
    span = x1 - x0
    for x in range(int(span)):
        t = x / max(span - 1, 1)
        col = tuple(int(r0[i] + (r1[i] - r0[i]) * t) for i in range(3))
        draw.rectangle([x0 + x, y0, x0 + x + 1, y1], fill=col)


def load_rows(board_path: Path) -> list[tuple[str, float]]:
    board = json.loads(board_path.read_text())
    rows = [
        (name, data["score"])
        for name, data in board["models"].items()
        if data.get("score") is not None
    ]
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def short(name: str) -> str:
    return {
        "Qwen3.5-4B (untrained)": "Qwen3.5-4B",
        "Laguna S 2.1 (free)": "Laguna S 2.1",
        "Nemotron 3 Ultra (free)": "Nemotron 3 Ultra",
    }.get(name, name)


def render_png(rows: list[tuple[str, float]], out: Path, scale: int = 2) -> None:
    H = canvas_height(len(rows))
    im = Image.new("RGB", (W * scale, H * scale), BG)
    d = ImageDraw.Draw(im)
    s = scale

    d.rounded_rectangle([40 * s, 24 * s, 1080 * s, (H - 24) * s], radius=14 * s,
                        fill=CARD, outline=EDGE, width=max(1, int(1.5 * s)))

    d.text((80 * s, TITLE_Y * s), "Benchmark scoreboard", font=font("serif", 43, s), fill=INK)

    d.line([80 * s, RULE_Y * s, 1040 * s, RULE_Y * s], fill=EDGE, width=max(1, s))
    hdr = font("mono", 11, s)
    # Centre RANK over its badge column; the tracked-out labels are wide enough
    # to collide with MODEL if they are both left-aligned.
    d.text((96 * s, HDR_Y * s), "R A N K", font=hdr, fill=FAINT, anchor="ma")
    d.text((142 * s, HDR_Y * s), "M O D E L", font=hdr, fill=FAINT)
    d.text((1040 * s, HDR_Y * s), "S C O R E", font=hdr, fill=FAINT, anchor="ra")

    for i, (name, score) in enumerate(rows):
        y = ROW_Y0 + i * ROW_STEP
        ours = name == OURS
        d.rounded_rectangle([80 * s, y * s, 112 * s, (y + 32) * s], radius=9 * s,
                            fill=BADGE[i] if i < 3 else BADGE_REST)
        d.text((96 * s, (y + 16) * s), str(i + 1), font=font("mono", 14, s),
               fill="#FFFFFF" if i in (0, 2) else "#2B454B", anchor="mm")

        label = short(name)
        d.text((142 * s, (y + 12) * s), label, font=font("sans", 19, s, bold=True),
               fill=INK, anchor="lm")
        tag = "TRAINED HERE" if ours else ("UNTRAINED BASE" if name == BASE else None)
        if tag:
            wlab = d.textlength(label, font=font("sans", 19, s, bold=True))
            d.text((142 * s + wlab + 14 * s, (y + 13) * s), tag, font=font("mono", 11, s),
                   fill=OURS_DARK if ours else FAINT, anchor="lm")
        d.text((1040 * s, (y + 12) * s), f"{score:.2f}", font=font("mono", 19, s),
               fill=INK, anchor="rm")

        ty0, ty1 = (y + 29) * s, (y + 41) * s
        d.rounded_rectangle([TRACK_X0 * s, ty0, TRACK_X1 * s, ty1], radius=6 * s, fill=TRACK_BG)
        bar_w = TRACK_W * s * (score / 100.0)
        if bar_w > 0:
            fill = Image.new("RGB", (max(int(bar_w), 1), int(ty1 - ty0)), TRACK_BG)
            hgrad(ImageDraw.Draw(fill), (0, 0, bar_w, ty1 - ty0),
                  OURS_DARK if ours else TEAL_DARK,
                  OURS_LIGHT if ours else TEAL_LIGHT, s)
            mask = Image.new("L", fill.size, 0)
            ImageDraw.Draw(mask).rounded_rectangle([0, 0, fill.size[0] - 1, fill.size[1] - 1],
                                                   radius=6 * s, fill=255)
            im.paste(fill, (TRACK_X0 * s, int(ty0)), mask)

    out.parent.mkdir(parents=True, exist_ok=True)
    im.save(out, "PNG", optimize=True)


def render_svg(rows: list[tuple[str, float]], out: Path) -> None:
    H = canvas_height(len(rows))
    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" aria-labelledby="title">',
        '  <title id="title">Live-Y benchmark scoreboard</title>',
        f'  <rect width="{W}" height="{H}" fill="{BG}"/>',
        f'  <rect x="40" y="24" width="1040" height="{H - 48}" rx="14" fill="{CARD}" '
        f'stroke="{EDGE}" stroke-width="1.5"/>',
        '  <defs>',
        f'    <linearGradient id="bar" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" '
        f'stop-color="{TEAL_DARK}"/><stop offset="100%" stop-color="{TEAL_LIGHT}"/></linearGradient>',
        f'    <linearGradient id="ours" x1="0" y1="0" x2="1" y2="0"><stop offset="0%" '
        f'stop-color="{OURS_DARK}"/><stop offset="100%" stop-color="{OURS_LIGHT}"/></linearGradient>',
        '  </defs>',
        f'  <text x="80" y="{TITLE_Y + 38}" fill="{INK}" font-family="Georgia, serif" '
        f'font-size="43">Benchmark scoreboard</text>',
        f'  <line x1="80" y1="{RULE_Y}" x2="1040" y2="{RULE_Y}" stroke="{EDGE}" stroke-width="1"/>',
        f'  <g fill="{FAINT}" font-family="IBM Plex Mono, Menlo, monospace" font-size="13" '
        f'font-weight="600" letter-spacing="1.6">'
        f'<text x="96" y="{HDR_Y + 10}" text-anchor="middle">RANK</text>'
        f'<text x="142" y="{HDR_Y + 10}">MODEL</text>'
        f'<text x="1040" y="{HDR_Y + 10}" text-anchor="end">SCORE</text></g>',
        '  <g font-family="IBM Plex Sans, Arial, sans-serif" fill="%s">' % INK,
    ]
    for i, (name, score) in enumerate(rows):
        y = ROW_Y0 + i * ROW_STEP
        ours = name == OURS
        badge = BADGE[i] if i < 3 else BADGE_REST
        num_fill = "#FFFFFF" if i in (0, 2) else "#2B454B"
        label = short(name)
        p.append(f'    <!-- {i+1} · {name} · {score:.2f} -->')
        p.append(f'    <rect x="80" y="{y}" width="32" height="32" rx="9" fill="{badge}"/>')
        p.append(f'    <text x="96" y="{y+21}" text-anchor="middle" fill="{num_fill}" '
                 f'font-family="IBM Plex Mono, Menlo, monospace" font-size="14" font-weight="700">{i+1}</text>')
        p.append(f'    <text x="142" y="{y+19}" font-size="19" font-weight="600">{label}</text>')
        tag = "TRAINED HERE" if ours else ("UNTRAINED BASE" if name == BASE else None)
        if tag:
            p.append(f'    <text x="{142 + 10 * len(label) + 14}" y="{y+19}" font-size="13" '
                     f'fill="{OURS_DARK if ours else FAINT}" '
                     f'font-family="IBM Plex Mono, Menlo, monospace" letter-spacing="1.4">{tag}</text>')
        p.append(f'    <text x="1040" y="{y+19}" text-anchor="end" fill="{INK}" '
                 f'font-family="IBM Plex Mono, Menlo, monospace" font-size="19">{score:.2f}</text>')
        p.append(f'    <rect x="{TRACK_X0}" y="{y+29}" width="{TRACK_W}" height="12" rx="6" fill="{TRACK_BG}"/>')
        p.append(f'    <rect x="{TRACK_X0}" y="{y+29}" width="{round(TRACK_W * score / 100)}" '
                 f'height="12" rx="6" fill="url(#{"ours" if ours else "bar"})"/>')
    p += ['  </g>', '</svg>', '']
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", type=Path, default=BOARD)
    ap.add_argument("--png", type=Path, default=PNG_OUT)
    ap.add_argument("--svg", type=Path, default=SVG_OUT)
    ap.add_argument("--scale", type=int, default=2)
    args = ap.parse_args()

    rows = load_rows(args.board)
    render_svg(rows, args.svg)
    render_png(rows, args.png, args.scale)
    for i, (name, score) in enumerate(rows, 1):
        print(f"  {i}. {name:26} {score:6.2f}")
    print(f"wrote {args.svg}")
    print(f"wrote {args.png}")


if __name__ == "__main__":
    main()
