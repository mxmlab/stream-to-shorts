#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Вертикальные клипы 1080x1920 из готовых нарезок.

Вход : <clips_root>\\**\\*.mov   (HEVC 1920x1080 60fps + AAC)
Выход: <clips_vertical_root>\\<та же структура>.mp4

Пути и шрифты — из config.json (см. config.py); раскладки вебки — из раздела
`layouts` конфига: имя -> {"cam": [w, h, x, y] в кадре 1920x1080, "game_x0": int}.

Запуск:
    python render_vertical.py            # пропускает уже готовые
    python render_vertical.py --force    # пересчитывает всё
    python render_vertical.py --only game2   # только один исходник
    python render_vertical.py --dry-run  # показать команды, ничего не рендерить
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import CONFIG  # noqa: E402

# ---------------------------------------------------------------- пути/константы

PY_DIR = os.path.dirname(os.path.abspath(__file__))
FFMPEG = CONFIG.ffmpeg
FFPROBE = CONFIG.ffprobe
FONT = CONFIG.font_caption

IN_ROOT = CONFIG.clips_root
OUT_ROOT = CONFIG.clips_vertical_root
OVERLAY = os.path.join(PY_DIR, "overlay.png")

OUT_W, OUT_H = 1080, 1920


def _load_layouts() -> dict:
    """Раздел `layouts` конфига -> {имя: ((cam_w, cam_h, cam_x, cam_y), game_x0)}."""
    out = {}
    for name, value in (CONFIG.layouts or {}).items():
        cam = (value or {}).get("cam")
        if not cam or len(cam) != 4:
            raise SystemExit("раскладка %r в конфиге: нужен ключ \"cam\": [w, h, x, y]" % name)
        out[name] = (tuple(int(v) for v in cam), int((value or {}).get("game_x0", 0)))
    return out


# исходник -> (CAM w:h:x:y в кадре 1920x1080, GAME_X0) — из config.json, раздел layouts
LAYOUT = _load_layouts()

GAME_W, GAME_H = 960, 1080          # crop игры (без вебки слева снизу и чата справа)
BG_SMALL = (270, 480)               # уменьшение перед размытием
BG_BLUR_SIGMA = 12
BG_BRIGHTNESS = 0.45                # затемнение до ~45 %

CAM_OUT = (1080, 608)               # вебка: scale 1080x608, в (0,0)
GAME_OUT = (1080, 1215)             # игра:  scale 1080x1215
GAME_Y = 612                        # ... в (0,612) — как в TASK.md
# ПРИМЕЧАНИЕ: в эталоне mock_A.jpg игра фактически лежит на y=604 (проверено
# корреляцией: 0.99954 при 604 против 0.93690 при 612), а полоса 604..611
# перекрывает её. Флаг --game-y 604 даёт раскладку ровно как в эталоне.

ENC = [
    "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0",
    "-profile:v", "high", "-pix_fmt", "yuv420p", "-r", "60",
    "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
    "-movflags", "+faststart",
]


# ---------------------------------------------------------------- overlay.png

def make_overlay(path: str = OVERLAY, text: str = "") -> str:
    """Оверлей 1080x1920 с прозрачностью: полоса-разделитель + плашка стрима.

    Плашка рисуется всегда; надпись — из `text` (пусто = без надписи). Так в
    открытом репозитории нет чужого ника: свой канал задаётся `--overlay-text`.
    """
    img = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # фиолетовая полоса-разделитель #9146FF, y 604..612 на всю ширину
    d.rectangle((0, 604, OUT_W - 1, 611), fill=(145, 70, 255, 255))

    f = ImageFont.truetype(FONT, 40)
    txt = text or ""
    tw = d.textlength(txt, font=f) if txt else 0
    pad_r = 96 if txt else 12
    x, y = 36, 520
    d.rounded_rectangle((x, y, x + tw + pad_r, y + 64), radius=32, fill=(145, 70, 255))
    gx, gy = x + 18, y + 14
    d.polygon([(gx,gy),(gx+30,gy),(gx+30,gy+22),(gx+20,gy+32),(gx+12,gy+32),(gx+6,gy+38),(gx+6,gy+32),(gx,gy+32)], fill="white")
    d.rectangle((gx+11,gy+8,gx+14,gy+18), fill=(145,70,255)); d.rectangle((gx+19,gy+8,gx+22,gy+18), fill=(145,70,255))
    if txt:
        d.text((x+62, y+8), txt, font=f, fill="white")

    img.save(path)
    return path


# ---------------------------------------------------------------- фильтры

def build_filter(cam: tuple, game_x0: int, game_y: int = GAME_Y) -> str:
    """Один filter_complex на клип: split -> три ветки -> overlay x3."""
    cw, ch, cx, cy = cam

    # затемнение до ~45 % яркости: colorlevels даёт мультипликативное
    # (значение 0.45 по каждому каналу rgb), в отличие от аддитивного eq=brightness.


    bg = (
        f"[0:v]crop={GAME_W}:{GAME_H}:{game_x0}:0,"
        f"scale={BG_SMALL[0]}:{BG_SMALL[1]},"
        f"gblur=sigma={BG_BLUR_SIGMA},"
        f"scale={OUT_W}:{OUT_H},"
        f"colorlevels=rimax={BG_BRIGHTNESS}:gimax={BG_BRIGHTNESS}:bimax={BG_BRIGHTNESS}"
        f"[bg]"
    )

    return (
        f"[0:v]split=3[g1][g2][g3];"
        f"{bg};"
        f"[g1]crop={cw}:{ch}:{cx}:{cy},scale={CAM_OUT[0]}:{CAM_OUT[1]}[cam];"
        f"[g2]crop={GAME_W}:{GAME_H}:{game_x0}:0,scale={GAME_OUT[0]}:{GAME_OUT[1]}[game];"
        f"[1:v]format=rgba,scale={OUT_W}:{OUT_H}[ov];"
        f"[bg][cam]overlay=0:0:format=auto:shortest=0[o1];"
        f"[o1][game]overlay=0:{game_y}:format=auto:shortest=0[o2];"
        # setsar=1: scale с разными множителями по осям иначе пишет SAR 128:81, и плееры сжимают кадр в 8:9
        f"[o2][ov]overlay=0:0:format=auto:shortest=0,setsar=1,format=yuv420p[v]"
        f";[g3]nullsink"
    )


# ---------------------------------------------------------------- утилиты

def source_of(rel: str) -> str:
    """Исходник = первая подпапка пути клипа."""
    return rel.split("/")[0]


def out_path_for(rel: str, out_root: str = None) -> str:
    base = rel[:-4] if rel.lower().endswith(".mov") else rel
    return os.path.join(out_root or OUT_ROOT, base.replace("/", os.sep) + ".mp4")


def list_clips() -> list:
    out = []
    for dirpath, _dirnames, filenames in os.walk(IN_ROOT):
        for fn in filenames:
            if fn.lower().endswith(".mov"):
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, IN_ROOT).replace(os.sep, "/")
                out.append((rel, full))
    out.sort(key=lambda t: t[0])
    return out


def probe_duration(path: str) -> float:
    r = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return -1.0


# ---------------------------------------------------------------- рендер

def render_one(src: str, dst: str, rel: str, dry: bool = False,
               game_y: int = GAME_Y) -> tuple:
    """Возвращает (ok, сообщение)."""
    s = source_of(rel)
    if s not in LAYOUT:
        return False, f"нет раскладки для исходника '{s}'"
    cam, game_x0 = LAYOUT[s]

    # Длительность = как у входа. Часть записей имеет нерегулярные метки времени
    # (первый кадр не с нуля, контейнер длиннее потока),
    # из-за чего CFR-нормализация добавляет ~0.13-0.15 с. Подрезаем выход по
    # длительности исходника; -0.001 с — чтобы муксер не выдал лишний кадр.
    src_dur = probe_duration(src)
    cap = ["-t", f"{src_dur - 0.001:.3f}"] if src_dur > 0 else []

    cmd = [
        FFMPEG, "-hide_banner", "-nostdin", "-y",
        "-i", src,
        "-i", OVERLAY,
        "-filter_complex", build_filter(cam, game_x0, game_y),
        "-map", "[v]", "-map", "0:a:0",
        *ENC,
        *cap,
        dst,
    ]
    if dry:
        print(" ".join(cmd))
        return True, "dry-run"

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(dst):
        tail = "\n".join((r.stderr or "").strip().splitlines()[-6:])
        if os.path.exists(dst):
            try:
                os.remove(dst)
            except OSError:
                pass
        return False, f"ffmpeg rc={r.returncode}\n{tail}"
    return True, f"{dt:.1f}s"


def main() -> int:
    # консоль Windows по умолчанию cp1251/cp866 — принудительно utf-8
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description="Вертикальные клипы 1080x1920 из нарезок")
    ap.add_argument("--force", action="store_true", help="пересчитать уже готовые")
    ap.add_argument("--only", default=None, help="только один исходник (первая подпапка)")
    ap.add_argument("--dry-run", action="store_true", help="не рендерить, показать команды")
    ap.add_argument("--limit", type=int, default=0, help="ограничить число клипов (для теста)")
    ap.add_argument("--out-root", default=None,
                    help=f"куда писать результат (по умолчанию {OUT_ROOT})")
    ap.add_argument("--game-y", type=int, default=GAME_Y,
                    help=f"y слоя игры (по умолчанию {GAME_Y}; в эталоне фактически 604)")
    ap.add_argument("--overlay-text", default="",
                    help="надпись на плашке overlay.png (пусто — плашка без надписи)")
    args = ap.parse_args()

    out_root = args.out_root or OUT_ROOT

    make_overlay(OVERLAY, args.overlay_text)

    clips = list_clips()
    if args.only:
        clips = [c for c in clips if source_of(c[0]) == args.only]
    if args.limit:
        clips = clips[: args.limit]

    total = len(clips)
    print(f"Оверлей: {OVERLAY}")
    print(f"Клипов к обработке: {total}")
    print(f"Куда пишем: {out_root}")
    print(f"Слой игры: y={args.game_y}")

    ok = 0
    skipped = 0
    errors = []
    t_start = time.time()

    for i, (rel, full) in enumerate(clips, 1):
        dst = out_path_for(rel, out_root)
        if os.path.exists(dst) and not args.force and not args.dry_run:
            skipped += 1
            print(f"[{i}/{total}] SKIP {rel}")
            continue

        good, msg = render_one(full, dst, rel, dry=args.dry_run, game_y=args.game_y)
        if good:
            ok += 1
            print(f"[{i}/{total}] OK   {rel}  ({msg})")
        else:
            errors.append((rel, msg))
            print(f"[{i}/{total}] FAIL {rel}  ({msg})")
        sys.stdout.flush()

    elapsed = time.time() - t_start
    print("-" * 60)
    print(f"Готово: {ok}   Пропущено (уже были): {skipped}   Ошибок: {len(errors)}")
    print(f"Время рендера: {elapsed:.1f} с = {elapsed/60:.1f} мин "
          f"({elapsed/max(ok,1):.1f} с на клип)")
    for rel, msg in errors:
        print(f"  FAIL {rel}: {msg.splitlines()[0]}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
