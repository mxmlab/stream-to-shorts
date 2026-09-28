#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Задание 4, часть C: слой субтитров/героя/заголовка на HyperFrames.

Запуск:
    python hf_overlay.py <edl.json> [--force] [--no-render]

Строит `hf\\<name>\\index.html` — композицию HyperFrames (прозрачный фон,
1080x1920, rail субтитров + hero + заголовок) — и рендерит
`hf\\<name>\\overlay.mov` (1080x1920, 60 fps, альфа; mov у HyperFrames всегда
ProRes 4444 / yuva444p10le) командой
`npx hyperframes render --format mov --fps 60 --gpu -o overlay.mov`.

Время в композиции — ВЫХОДНОЕ время ролика: слова/сегменты считает Plan из
render_tt.py (импорт), своя логика пересчёта не дублируется.

Проверки (таблица; код возврата 1 при провале):
  1. overlay.mov: 1080x1920, 60 fps, есть альфа (pix_fmt с 'a'),
     длительность = длительности ролика ±1 кадр;
  2. `npx hyperframes lint` по проекту — без ошибок;
  3. кадр в момент середины каждого героя и каждого 5-го слова: непрозрачные
     пиксели есть в полосе caption_y ±120; в момент, когда по плану ничего не
     видно (паузы > 0.6 с без групп), — прозрачно в этой полосе.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

TT_DIR = os.path.dirname(os.path.abspath(__file__))
if TT_DIR not in sys.path:
    sys.path.insert(0, TT_DIR)
import render_tt as rt                                 # noqa: E402

HF_DIR = os.path.join(TT_DIR, "hf")
HF_CLI = os.path.join(HF_DIR, "node_modules", "hyperframes", "bin", "hyperframes.mjs")
GSAP_JS = os.path.join(HF_DIR, "node_modules", "gsap", "dist", "gsap.min.js")
NPX = shutil.which("npx") or "npx"
NODE = shutil.which("node") or "node"
# кэш npm — внутри проекта: штатный пользовательский кэш может быть недоступен
# на запись, и npx падает с EPERM ещё до запуска CLI
CLI_ENV = {**os.environ, "npm_config_cache": os.path.join(HF_DIR, ".npmcache")}

# Шрифты — из конфига (ключи font_caption / font_title, папка fonts_dir).
# FONTS — пары (имя семейства для CSS @font-face, имя файла в fonts_dir).
FONT_DIR = rt.FONTS_DIR
FONTS = (("Montserrat ExtraBold", os.path.basename(rt.FONT)),
         ("Montserrat Black", os.path.basename(rt.FONT_TITLE)))
FONT_EXTRA = os.path.join(FONT_DIR, FONTS[0][1])
FONT_BLACK = os.path.join(FONT_DIR, FONTS[1][1])

W, H, FPS = 1080, 1920, 60

# --- rail (строка субтитров)
RAIL_SIZE = 70
RAIL_MAX_W = 860
RAIL_LEAD = 1.20                  # межстрочный, px на строку = RAIL_SIZE*RAIL_LEAD
RAIL_IN = 0.08                    # контейнер показывается за 0.08 с до первого слова
RAIL_WORD_DUR = 0.22
RAIL_OUT_DUR = 0.15
RAIL_TAIL = 0.5                   # группа живёт +0.5 с после последнего слова
RAIL_MAX_WORDS = 5
RAIL_MAX_DUR = 2.2
RAIL_GAP = 0.30                   # пауза, после которой начинается новая группа
YELLOW = "#FFD84D"
EMPH_SCALE = 1.12
EMPH_DUR = 0.20

# --- hero
HERO_MAX_SIZE = 150
HERO_MAX_W = 940
HERO_LEAD = 1.10

# --- заголовок (геометрия — как в render_tt.make_title_png)
TITLE_TOP = rt.TITLE_TOP
TITLE_IN_T = 0.15
TITLE_IN_DUR = 0.40
TITLE_OUT_DUR = 0.25
TITLE_UNTIL_DEFAULT = 2.8

PARASITES = ("э", "ээ", "эм", "ммм")     # не показываем никогда
PARASITE_NU = "ну"                       # скрываем, если первое в группе и группа не из одного слова

BAND = 120                               # полоса проверки вокруг caption_y, px
ALPHA_OPAQUE = 32                        # «непрозрачные пиксели есть»
LEAD = 0.05                              # допуск сопоставления EDL-времени, с


def log(msg: str = "") -> None:
    print(msg, flush=True)


def font(size: int, black: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BLACK if black else FONT_EXTRA, size)


_TMPD = ImageDraw.Draw(Image.new("RGBA", (8, 8)))


def text_w(text: str, f) -> float:
    return _TMPD.textlength(text, font=f)


# ================================================================ расписание слоя


def build_schedule(plan, edl: dict, bad, ok) -> dict:
    """Rail-группы, герои и заголовок в ВЫХОДНОМ времени ролика."""
    drop = [float(x) for x in (edl.get("drop") or [])]
    emph = [float(x) for x in (edl.get("emph") or [])]

    words = []
    for w in plan.words:
        if any(abs(w["src_s"] - d) <= LEAD for d in drop):
            continue                      # EDL drop — слово не показываем
        x = dict(w)
        x["emph"] = any(abs(w["src_s"] - e) <= LEAD for e in emph)
        words.append(x)

    # --- группы: 5 слов / 2.2 с / пауза 0.30 с / граница сегмента
    groups, cur = [], []
    for w in words:
        if cur:
            gap = w["s"] - cur[-1]["e"]
            dur = w["e"] - cur[0]["s"]
            if (w["seg"] != cur[-1]["seg"] or gap > RAIL_GAP
                    or len(cur) >= RAIL_MAX_WORDS or dur > RAIL_MAX_DUR):
                groups.append(cur)
                cur = []
        cur.append(w)
    if cur:
        groups.append(cur)

    # --- группа из 1 слова — только отдельная реплика (соседи дальше 0.30 с)
    kept = []
    for gi, g in enumerate(groups):
        if len(g) == 1:
            prev_e = groups[gi - 1][-1]["e"] if gi > 0 else -1e9
            next_s = groups[gi + 1][0]["s"] if gi + 1 < len(groups) else 1e9
            if not (g[0]["s"] - prev_e > RAIL_GAP and next_s - g[0]["e"] > RAIL_GAP):
                continue
        # --- слова-паразиты: «э/ээ/эм/ммм» всегда, «ну» — первым в группе
        ws = []
        for k, w in enumerate(g):
            t = w["w"].strip().lower()
            if t in PARASITES:
                continue
            if t == PARASITE_NU and k == 0 and len(g) > 1:
                continue
            ws.append(w)
        if ws:
            kept.append(ws)
    groups = kept

    # --- герои (время ИСХОДНИКА + номер сегмента -> выходное время)
    heroes = []
    for h in (edl.get("hero") or []):
        si = int(h["seg"])
        if not (0 <= si < len(plan.segments)):
            raise ValueError(f"hero: нет сегмента {si}")
        seg = plan.segments[si]
        t0 = seg["out"] + (float(h["from"]) - seg["a"]) / seg["speed"]
        t1 = seg["out"] + (float(h["to"]) - seg["a"]) / seg["speed"]
        t0 = min(max(t0, seg["out"]), seg["out"] + seg["dur"])
        t1 = min(max(t1, seg["out"]), seg["out"] + seg["dur"])
        if t1 - t0 <= 0:
            raise ValueError(f"hero seg={si}: пустое окно {t0}..{t1}")
        heroes.append({"start": t0, "end": t1,
                       "text": str(h["text"]), "seg": si,
                       "lines": hero_lines(str(h["text"]))})

    # --- группы, попавшие в окно героя, не показываются
    if heroes:
        groups = [g for g in groups
                  if not any(_overlap(g[0]["s"], g[-1]["e"], h["start"], h["end"])
                             for h in heroes)]

    # --- rail: интервалы показа + строки переноса
    rail_font = font(RAIL_SIZE)
    sched_groups = []
    for gi, g in enumerate(groups):
        s0, e0 = g[0]["s"], g[-1]["e"]
        nxt = groups[gi + 1][0]["s"] - RAIL_IN if gi + 1 < len(groups) else None
        out = e0 + RAIL_TAIL
        if nxt is not None:
            out = min(out, nxt)
        out = max(out, g[-1]["s"])               # минимум — до начала последнего слова
        lines = rail_lines(g, rail_font)
        h = RAIL_SIZE * RAIL_LEAD * len(lines)
        sched_groups.append({
            "words": g,
            "in": max(0.0, s0 - RAIL_IN),
            "out": out,
            "end": out + RAIL_OUT_DUR,
            "lines": lines,
            "top": plan.caption_y - h / 2.0,
            "height": h,
        })

    # --- заголовок
    title = None
    if plan.title:
        size, lines = rt._title_layout(plan.title)
        f = font(size)
        tmp = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
        boxes = [tmp.textbbox((0, 0), ln, font=f) for ln in lines]
        widths = [b[2] - b[0] for b in boxes]
        cap_h = -tmp.textbbox((0, 0), "Н", font=f, anchor="ls")[1]
        asc, desc = f.getmetrics()
        step = size * rt.TITLE_LINE_SPACING
        text_w_ = max(widths)
        pw = int(round(text_w_)) + 2 * rt.TITLE_PAD_X
        ph = round(cap_h + (len(lines) - 1) * step) + 2 * rt.TITLE_PAD_Y
        # верх строки в HTML: базовая линия i = PAD_Y + cap_h + i*step от верха плашки
        tops = [rt.TITLE_PAD_Y + cap_h + i * step
                - ((step - (asc + desc)) / 2.0 + asc) for i in range(len(lines))]
        until = edl.get("title_until", TITLE_UNTIL_DEFAULT)
        until = float(until)
        title = {"size": size, "lines": lines, "pw": pw, "ph": ph,
                 "left": (W - pw) // 2, "tops": [round(t, 3) for t in tops],
                 "step": step, "text_w": text_w_,
                 "until": until,
                 "end": plan.total if until < 0 else until + TITLE_OUT_DUR}

    # --- плоский список показанных слов (для проверки «каждого 5-го»)
    rail_words = []
    for g in sched_groups:
        for w in g["words"]:
            rail_words.append({"w": w["w"], "s": w["s"], "e": w["e"],
                               "group": g})

    return {"groups": sched_groups, "heroes": heroes, "title": title,
            "rail_words": rail_words, "total": plan.total,
            "caption_y": plan.caption_y}


def _overlap(a0, a1, b0, b1) -> bool:
    return a1 > b0 and b1 > a0


def rail_lines(ws: list, f) -> list:
    """Перенос группы по словам: ширина блока <= RAIL_MAX_W, максимум 2 строки."""
    lines, cur = [], []
    for w in ws:
        trial = cur + [w]
        if cur and text_w(" ".join(_plain(x) for x in trial), f) > RAIL_MAX_W:
            lines.append(cur)
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(cur)
    while len(lines) > 2:                      # максимум 2 строки
        lines[-2].extend(lines[-1])
        lines.pop()
    return lines


def _plain(w: dict) -> str:
    return w.get("disp", w["w"])


def hero_lines(text: str) -> list:
    """Размер и строки героя: самая длинная строка <= HERO_MAX_W, размер <= 150.

    Сначала максимум размера, при равенстве — меньше строк; максимум 2 строки.
    """
    words = text.upper().split()
    if not words:
        return {"size": HERO_MAX_SIZE, "lines": [""]}
    cands = [[text.upper()]]
    for k in range(1, len(words)):
        cands.append([" ".join(words[:k]), " ".join(words[k:])])
    best = None
    for lines in cands:
        if len(lines) > 2:
            continue
        size = HERO_MAX_SIZE
        while size > 24:
            f = font(size, black=True)
            if max(text_w(ln, f) for ln in lines) <= HERO_MAX_W:
                break
            size -= 1
        if size <= 24:
            continue
        key = (size, -len(lines))
        if best is None or key > best[0]:
            best = (key, size, lines)
    if best is None:                           # не влезло даже 24 px — как есть
        return {"size": 24, "lines": words}
    return {"size": best[1], "lines": best[2]}


# ================================================================ HTML


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def build_html(plan, edl: dict, sched: dict, bad, ok) -> str:
    name = plan.name
    out = []
    out.append("<!doctype html>")
    out.append('<html lang="ru">')
    out.append("<head>")
    out.append('<meta charset="UTF-8" />')
    out.append('<meta name="viewport" content="width=1080, height=1920" />')
    out.append(f"<title>{_esc(name)} — HyperFrames overlay</title>")
    out.append("<style>")
    out.append("".join(
        f"@font-face {{ font-family: '{fam}'; "
        f"src: url('{fn}') format('truetype'); font-weight: normal; "
        f"font-style: normal; }}\n" for fam, fn in FONTS))
    out.append("""
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { width: 1080px; height: 1920px; overflow: hidden; background: transparent; }
#root { position: relative; width: 100%; height: 100%; overflow: hidden; }
.clip { position: absolute; }
.grp { left: 0; width: 1080px; }
.g-in { position: absolute; left: 0; top: 0; width: 1080px; height: 100%; }
/* между span слов лежит настоящий пробельный текстовый узел (см. build_html):
   пробел должен доезжать до кадра, поэтому white-space: normal, а не nowrap.
   Шрифт и размер у .line те же, что у .wd: пробел — отдельный текстовый узел,
   он наследует шрифт от .line, а не от соседнего span'а; без font-size он
   рисовался бы браузерными 16 px (пробел 4 px вместо 20 px при 70 px). */
.line { display: block; text-align: center; white-space: normal;
        font-family: 'Montserrat ExtraBold'; font-size: 70px; }
.wd { display: inline-block; font-family: 'Montserrat ExtraBold'; font-size: 70px;
      line-height: 1; color: #ffffff; -webkit-text-stroke: 3px rgba(0,0,0,.55);
      paint-order: stroke fill; text-shadow: 0 4px 18px rgba(0,0,0,.55); }
.wd.emph { color: #FFD84D; }
.hero { left: 0; width: 1080px; }
.h-in { position: absolute; left: 0; top: 0; width: 1080px; height: 100%;
        text-align: center; }
.h-line { display: block; white-space: nowrap; font-family: 'Montserrat Black';
          color: #ffffff; -webkit-text-stroke: 6px #000000;
          paint-order: stroke fill; text-shadow: 0 8px 30px rgba(0,0,0,.6); }
.ttl { left: 0; top: 0; width: 1080px; height: 1920px; }
.ttl-plate { position: absolute; background: #ffffff; border-radius: 22px;
             overflow: hidden; }
.ttl-line { position: absolute; text-align: center; color: #000000;
            font-family: 'Montserrat ExtraBold'; white-space: nowrap; }
""")
    out.append("</style>")

    out.append("</head>")
    out.append("<body>")
    out.append(f'<div id="root" data-composition-id="{_esc(name)}" data-start="0" '
               f'data-width="{W}" data-height="{H}" '
               f'data-duration="{sched["total"]:.6f}">')

    js = ["window.__timelines = window.__timelines || {};",
          "const tl = gsap.timeline({ paused: true });"]

    # --- rail
    for gi, g in enumerate(sched["groups"]):
        dur = g["end"] - g["in"]
        out.append(f'  <div class="clip grp" id="g{gi}" data-start="{g["in"]:.4f}" '
                   f'data-duration="{dur:.4f}" data-track-index="0" '
                   f'style="top:{g["top"]:.2f}px;height:{g["height"]:.2f}px">')
        out.append(f'    <div class="g-in" id="g{gi}in">')
        for li, line in enumerate(g["lines"]):
            idx = {id(x): k for k, x in enumerate(g["words"])}
            out.append(f'      <div class="line" style="height:'
                       f'{RAIL_SIZE * RAIL_LEAD:.2f}px;line-height:'
                       f'{RAIL_SIZE * RAIL_LEAD:.2f}px">')
            spans = []
            for w in line:
                gi_w = idx[id(w)]
                txt = rt.censor(w["w"], bad, ok)
                if g["words"][0] is w and txt:       # первая буква группы — заглавная
                    txt = txt[0].upper() + txt[1:]
                cls = "wd emph" if w["emph"] else "wd"
                spans.append(f'<span class="{cls}" id="w{gi}_{gi_w}">'
                             f'{_esc(txt)}</span>')
            # слова строки — inline-block span'ы, поэтому между ними должен
            # стоять НАСТОЯЩИЙ пробел (текстовый узел вне span), иначе в кадре
            # слова слипаются («Б*яятакая», «Равняется12»); у .line при этом
            # white-space: normal, чтобы пробел не схлопывался
            out.append("        " + " ".join(spans))
            out.append("      </div>")
        out.append("    </div>")
        out.append("  </div>")

        js.append(f'tl.set("#g{gi}in", {{ opacity: 0, y: 0 }}, 0);')
        js.append(f'tl.set("#g{gi}in", {{ opacity: 1 }}, {g["in"]:.4f});')
        for wi, w in enumerate(g["words"]):
            sid = f"#w{gi}_{wi}"
            js.append(f'tl.set("{sid}", {{ opacity: 0, y: 10, scale: 1 }}, 0);')
            js.append(f'tl.fromTo("{sid}", {{ opacity: 0, y: 10 }}, '
                      f'{{ opacity: 1, y: 0, duration: {RAIL_WORD_DUR}, '
                      f'ease: "power2.out" }}, {w["s"]:.4f});')
            if w["emph"]:
                js.append(f'tl.fromTo("{sid}", {{ scale: {EMPH_SCALE} }}, '
                          f'{{ scale: 1, duration: {EMPH_DUR}, '
                          f'ease: "back.out(1.6)" }}, {w["s"]:.4f});')
        js.append(f'tl.to("#g{gi}in", {{ opacity: 0, y: -6, duration: '
                  f'{RAIL_OUT_DUR}, ease: "power2.in" }}, {g["out"]:.4f});')

    # --- hero
    for hi, h in enumerate(sched["heroes"]):
        lay = h["lines"]
        lh = lay["size"] * HERO_LEAD
        block = lh * len(lay["lines"])
        top = sched["caption_y"] - block / 2.0
        out.append(f'  <div class="clip hero" id="h{hi}" data-start="{h["start"]:.4f}" '
                   f'data-duration="{h["end"] - h["start"]:.4f}" data-track-index="1" '
                   f'style="top:{top:.2f}px;height:{block:.2f}px">')
        out.append(f'    <div class="h-in" id="h{hi}in">')
        for ln in lay["lines"]:
            out.append(f'      <div class="h-line" style="font-size:{lay["size"]}px;'
                       f'height:{lh:.2f}px;line-height:{lh:.2f}px">'
                       f'{_esc(ln)}</div>')
        out.append("    </div>")
        out.append("  </div>")

        js.append(f'tl.set("#h{hi}in", {{ opacity: 0, scale: 1.3, rotate: 1.2 }}, 0);')
        js.append(f'tl.fromTo("#h{hi}in", {{ opacity: 0, scale: 1.3, rotate: 1.2 }}, '
                  f'{{ opacity: 1, scale: 1, rotate: 0, duration: 0.18, '
                  f'ease: "back.out(1.7)" }}, {h["start"]:.4f});')
        js.append(f'tl.to("#h{hi}in", {{ opacity: 0, scale: 0.96, duration: 0.14, '
                  f'ease: "power3.in" }}, {h["end"] - 0.14:.4f});')

    # --- заголовок
    t = sched["title"]
    if t:
        out.append(f'  <div class="clip ttl" id="t0" data-start="0" '
                   f'data-duration="{t["end"]:.4f}" data-track-index="2">')
        out.append(f'    <div class="ttl-plate" id="t0plate" style="left:{t["left"]}px;'
                   f'top:{TITLE_TOP}px;width:{t["pw"]}px;height:{t["ph"]}px">')
        for i, ln in enumerate(t["lines"]):
            out.append(f'      <div class="ttl-line" style="left:{rt.TITLE_PAD_X}px;'
                       f'top:{t["tops"][i]:.2f}px;width:{t["text_w"]:.0f}px;'
                       f'height:{t["step"]:.2f}px;line-height:{t["step"]:.2f}px;'
                       f'font-size:{t["size"]}px">{_esc(ln)}</div>')
        out.append("    </div>")
        out.append("  </div>")

        js.append('tl.set("#t0plate", { opacity: 1, y: 0, '
                  'clipPath: "inset(0 100% 0 0)" }, 0);')
        js.append('tl.fromTo("#t0plate", { clipPath: "inset(0 100% 0 0)" }, '
                  '{ clipPath: "inset(0 0 0 0)", duration: '
                  f'{TITLE_IN_DUR}, ease: "expo.out" }}, {TITLE_IN_T});')
        if t["until"] >= 0:
            js.append('tl.to("#t0plate", { y: -12, opacity: 0, duration: '
                      f'{TITLE_OUT_DUR}, ease: "power2.in" }}, {t["until"]:.4f});')

    js.append(f'window.__timelines["{name}"] = tl;')
    js.append("tl.seek(0);")

    out.append("</div>")
    out.append('<script src="gsap.min.js"></script>')
    out.append("<script>")
    out.extend(js)
    out.append("</script>")
    out.append("</body>")
    out.append("</html>")
    return "\n".join(out) + "\n"


def write_if_changed(path: str, text: str) -> bool:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            if fh.read() == text:
                return False
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return True


def copy_if_needed(src: str, dst: str) -> None:
    if not os.path.exists(src):
        raise RuntimeError(f"нет файла {src}")
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        return
    shutil.copyfile(src, dst)


def project_dir(plan) -> str:
    return os.path.join(HF_DIR, plan.name)


# ================================================================ рендер


def run_cli(args: list, cwd: str) -> subprocess.CompletedProcess:
    """`npx hyperframes <args>`; если npx не может запуститься — bin/hyperframes.mjs."""
    try:
        r = subprocess.run([NPX, "hyperframes", *args], cwd=cwd, env=CLI_ENV,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        bad = r.returncode == 127 or "npm error" in (r.stderr or "")
    except OSError:
        r, bad = None, True
    if bad:
        log(f"  npx не запустился — вызываю {HF_CLI} напрямую")
        r = subprocess.run([NODE, HF_CLI, *args], cwd=cwd, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
    return r


def lint(proj: str) -> tuple:
    """(ошибок, предупреждений, текст)."""
    r = run_cli(["lint", "--json"], proj)
    txt = (r.stdout or "") + (r.stderr or "")
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return int(d.get("errorCount", -1)), int(d.get("warningCount", -1)), txt
        except ValueError:
            pass
    e = re.search(r"(\d+)\s+errors?,\s+(\d+)\s+warnings?", txt)
    if e:
        return int(e.group(1)), int(e.group(2)), txt
    return -1, -1, txt


def render(proj: str, force: bool, do_render: bool, inputs: list = ()) -> tuple:
    """(путь к overlay.mov, секунд, вывод, рендер удался).

    Готовый overlay.mov переиспользуется, если он новее EDL, `.words.json`,
    index.html и gsap.min.js (--force — перерендерить).
    """
    mov = os.path.join(proj, "overlay.mov")
    if not force and os.path.exists(mov):
        newest = max([os.path.getmtime(os.path.join(proj, "index.html")),
                      os.path.getmtime(os.path.join(proj, "gsap.min.js"))]
                     + [os.path.getmtime(p) for p in inputs if os.path.exists(p)])
        if os.path.getmtime(mov) > newest:
            return mov, 0.0, f"{os.path.basename(mov)} новее входа — рендер пропущен", True
    if not do_render:
        return mov, 0.0, "рендер отключён (--no-render)", os.path.exists(mov)
    t0 = time.time()
    r = run_cli(["render", "--format", "mov", "--fps", str(FPS), "--gpu",
                 "-o", "overlay.mov"], proj)
    dt = time.time() - t0
    ok = (r.returncode == 0 and os.path.exists(mov)
          and os.path.getmtime(mov) >= t0)      # старый файл за свежий не считаем
    return mov, dt, (r.stdout or "") + (r.stderr or ""), ok


# ================================================================ проверки


def probe_mov(path: str) -> dict:
    r = rt.run([rt.FFPROBE, "-hide_banner", "-v", "error", "-select_streams", "v:0",
                "-show_entries",
                "stream=width,height,pix_fmt,r_frame_rate,nb_frames,duration",
                "-show_entries", "format=duration", "-of", "json", path])
    try:
        d = json.loads(r.stdout)
        st = d["streams"][0]
        st["duration_fmt"] = d.get("format", {}).get("duration")
        return st
    except Exception:                                     # noqa: BLE001
        return {}


def band_max_alpha(path: str, t: float, y0: int, y1: int) -> int:
    """Максимум альфы в полосе строк y0..y1 кадра в момент t (-1 — кадра нет)."""
    r = subprocess.run([rt.FFMPEG, "-hide_banner", "-nostdin", "-v", "error",
                        "-ss", f"{max(t, 0.0):.4f}", "-i", path, "-frames:v", "1",
                        "-pix_fmt", "rgba", "-f", "rawvideo", "-"],
                       capture_output=True)
    buf = r.stdout or b""
    need = W * H * 4
    if len(buf) < need:
        return -1
    a = np.frombuffer(buf[:need], dtype=np.uint8).reshape(H, W, 4)
    y0 = max(0, min(H - 1, y0))
    y1 = max(y0 + 1, min(H, y1))
    return int(a[y0:y1, :, 3].max())


def merge_intervals(iv: list) -> list:
    iv = sorted(x for x in iv if x[1] > x[0])
    out = []
    for s, e in iv:
        if out and s <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def verify(sched: dict, mov: str, lint_res: tuple, proj: str) -> list:
    rows = []

    def add(name, expected, got, ok):
        rows.append((name, expected, got, bool(ok)))

    # 1. overlay.mov
    st = probe_mov(mov)
    total = sched["total"]
    add("overlay.mov: кадр", "1080x1920",
        f"{st.get('width')}x{st.get('height')}" if st else "нет файла",
        bool(st) and st.get("width") == W and st.get("height") == H)
    add("overlay.mov: частота кадров", "60/1", str(st.get("r_frame_rate")),
        st.get("r_frame_rate") == "60/1")
    pix = str(st.get("pix_fmt") or "")
    add("overlay.mov: альфа", "pix_fmt содержит 'a'", pix or "—",
        "a" in pix.lower())
    try:
        dur = float(st.get("duration") or st.get("duration_fmt"))
    except (TypeError, ValueError):
        dur = -1.0
    add("overlay.mov: длительность", f"{total:.3f} ±{1/FPS:.3f} с (1 кадр)",
        f"{dur:.3f}" if dur > 0 else "—", dur > 0 and abs(dur - total) <= 1.5 / FPS)

    # 2. lint
    errors, warns, _txt = lint_res
    add("npx hyperframes lint", "0 ошибок", f"{errors} ошибок, {warns} предупреждений",
        errors == 0)

    # 3. кадры
    y0, y1 = sched["caption_y"] - BAND, sched["caption_y"] + BAND
    hero_mid = [(h["start"] + h["end"]) / 2.0 for h in sched["heroes"]]
    word_pick = sched["rail_words"][::5]
    word_t = []
    for w in word_pick:
        g = w["group"]
        mid = (w["s"] + w["e"]) / 2.0
        mid = min(mid, g["out"] - 0.005)          # не попадать в затухание группы
        if mid >= w["s"]:
            word_t.append(mid)
    bad_hero, bad_word = [], []
    for t in hero_mid:
        a = band_max_alpha(mov, t, y0, y1)
        if a < ALPHA_OPAQUE:
            bad_hero.append(f"{t:.2f}с:a={a}")
    for t in word_t:
        a = band_max_alpha(mov, t, y0, y1)
        if a < ALPHA_OPAQUE:
            bad_word.append(f"{t:.2f}с:a={a}")
    add(f"герой: непрозрачно в полосе y {y0}..{y1} ({len(hero_mid)} шт)",
        "во всех кадрах есть альфа",
        "все есть" if not bad_hero else "нет: " + ", ".join(bad_hero[:4]),
        not bad_hero and bool(hero_mid))
    add(f"каждое 5-е слово: непрозрачно в полосе ({len(word_t)} шт)",
        "во всех кадрах есть альфа",
        "все есть" if not bad_word else "нет: " + ", ".join(bad_word[:4]),
        not bad_word and bool(word_t))

    # паузы > 0.6 с без групп — полоса должна быть прозрачной
    busy = [[g["in"], g["end"]] for g in sched["groups"]]
    busy += [[h["start"], h["end"]] for h in sched["heroes"]]
    busy = merge_intervals([iv for iv in busy if iv[0] < total])
    gaps, prev = [], 0.0
    for s, e in busy:
        if s - prev > 0.6:
            gaps.append((prev, s))
        prev = max(prev, e)
    if total - prev > 0.6:
        gaps.append((prev, total))
    bad_gap = []
    for s, e in gaps:
        t = (s + e) / 2.0
        a = band_max_alpha(mov, t, y0, y1)
        if a > 0:
            bad_gap.append(f"{t:.2f}с:a={a}")
    add(f"паузы > 0.6 с ({len(gaps)} шт): прозрачно",
        "альфа = 0 во всей полосе",
        "все прозрачны" if not bad_gap else "не прозрачно: " + ", ".join(bad_gap[:4]),
        not bad_gap)
    return rows


# ================================================================ запуск


def process(edl_path: str, force: bool = False, do_render: bool = True) -> bool:
    t_all = time.time()
    with open(edl_path, encoding="utf-8") as fh:
        edl = json.load(fh)
    bad, ok = rt.load_stems()
    words, words_src = rt.load_words_for(edl, edl_path)
    plan = rt.Plan(edl, words, bad, ok)
    sched = build_schedule(plan, edl, bad, ok)
    proj = project_dir(plan)
    os.makedirs(proj, exist_ok=True)
    sc = rt.sidecar_words(edl, edl_path)
    words_path = sc[1] if sc else os.path.join(rt.ANALYSIS_DIR, edl["src"], "words.json")

    html = build_html(plan, edl, sched, bad, ok)
    changed = write_if_changed(os.path.join(proj, "index.html"), html)
    copy_if_needed(GSAP_JS, os.path.join(proj, "gsap.min.js"))
    for fam, fn in FONTS:
        copy_if_needed(os.path.join(FONT_DIR, fn), os.path.join(proj, fn))

    log("=" * 78)
    log(f"HF-слой: {edl_path}")
    log(f"  источник слов: {words_src} — {len(words)} шт")
    log(f"  проект: {proj}  (index.html {'перезаписан' if changed else 'без изменений'})")
    log(f"  длительность ролика {plan.total:.3f} с, caption_y={plan.caption_y}, "
        f"групп={len(sched['groups'])}  героев={len(sched['heroes'])}  "
        f"заголовок={'есть' if sched['title'] else 'нет'}"
        + (f" (до {sched['title']['until']:.2f} с)" if sched["title"] else ""))
    for h in sched["heroes"]:
        log(f"  герой seg={h['seg']}: {h['start']:.2f}..{h['end']:.2f} с, "
            f"размер {h['lines']['size']} px, строк {len(h['lines']['lines'])}: "
            + " / ".join(h["lines"]["lines"]))

    mov, dt, rlog, rok = render(proj, force, do_render, [edl_path, words_path])
    tail = "\n".join((rlog or "").strip().splitlines()[-4:])
    if do_render and dt:
        log(f"  рендер HyperFrames: {dt:.1f} с")
    else:
        log(f"  {rlog.strip().splitlines()[0] if rlog.strip() else 'рендер не запускался'}")
    if do_render and not rok:
        log(tail)
        raise RuntimeError(f"overlay.mov не собрался:\n{tail}")

    lint_res = lint(proj)
    try:
        rows = verify(sched, mov, lint_res, proj)
    except Exception as exc:                                # noqa: BLE001
        raise RuntimeError(f"проверки слоя не прошли: {exc}") from exc
    rt.print_table(f"hf_overlay: {plan.name}", rows)
    log(f"  файл слоя: {mov}")
    log(f"  всего на слой: {time.time() - t_all:.1f} с")
    return all(r[3] for r in rows)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description="Слой субтитров/заголовка на HyperFrames")
    ap.add_argument("edl", nargs="+", help="монтажный лист (JSON)")
    ap.add_argument("--force", action="store_true", help="перерендерить overlay.mov")
    ap.add_argument("--no-render", action="store_true",
                    help="только собрать проект и проверить готовый overlay.mov")
    args = ap.parse_args()

    all_ok = True
    for edl_path in args.edl:
        try:
            ok = process(edl_path, force=args.force, do_render=not args.no_render)
        except Exception as exc:                            # noqa: BLE001
            log(f"ОШИБКА на {edl_path}: {exc}")
            ok = False
        all_ok = all_ok and ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
