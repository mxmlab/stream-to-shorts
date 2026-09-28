#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Движок динамичного монтажа вертикалок для TikTok (TASK.md в этой же папке).

Вход : монтажный лист (EDL) JSON — см. examples/edl_clip.json и README.
Выход: out\\<name>.mp4 (1080x1920, 60 fps, SAR 1:1, h264_nvenc cq 18, AAC 192k 48k).

Запуск:
    python render_tt.py <edl.json> [<edl2.json> ...] [--sheet] [--keep-temp]

Раскладка кадра берётся из render_vertical.py (LAYOUT из раздела `layouts`
конфига, build_filter, OVERLAY) — соседний модуль; шрифты, словари мата,
ffmpeg и папки приходят из config.json.

# Adapted from Reelsi (https://github.com/mxmlab/reelsi), AGPL-3.0, reelsi/core/censor.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- пути

TT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(TT_DIR))      # корень репозитория
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from config import CONFIG              # noqa: E402
from sources import find_source        # noqa: E402  (единый поиск исходника)
import render_vertical as rv          # noqa: E402  (LAYOUT / build_filter / OVERLAY)

FFMPEG = CONFIG.ffmpeg
FFPROBE = CONFIG.ffprobe

STREAM_ROOT = CONFIG.stream_root
ANALYSIS_DIR = CONFIG.analysis_dir

OUT_DIR = CONFIG.out_dir
TEMP_ROOT = CONFIG.temp_dir
BOOM_WAV = CONFIG.boom_wav

FONT = CONFIG.font_caption
FONT_TITLE = CONFIG.font_title
FONTS_DIR = CONFIG.fonts_dir
BADWORDS_TXT = CONFIG.badwords_path
OKWORDS_TXT = CONFIG.okwords_path

SR = 48000

# ---------------------------------------------------------------- константы кадра

OUT_W, OUT_H = 1080, 1920
SRC_W, SRC_H = 1920, 1080         # кадр исходника (для game_view)

# Панели для punch-зума (x всегда 0, ширина всегда 1080):
CAM_PANEL = (1080, 608, 0)        # вебка: y 0..607
GAME_PANEL = (1080, 1215, 612)    # игра: y 612..1826

CAM_ALT_SCALE = 1.07              # cam_alt: зум вебки в нечётных сегментах
# 1080x608 * 1.07, чётные стороны (yuv420p) — из них центр режется обратно
CAM_ALT_W = 2 * int(round(CAM_PANEL[0] * CAM_ALT_SCALE / 2))
CAM_ALT_H = 2 * int(round(CAM_PANEL[1] * CAM_ALT_SCALE / 2))

PUNCH_ATTACK = 0.07               # с, разгон зума
PUNCH_RELEASE = 0.25              # с, возврат к 1 в конце

# ---------------------------------------------------------------- константы субтитров

SUB_SIZE = 84
SUB_OUTLINE = 7
SUB_POS = (540, 1300)             # \pos по умолчанию (\an5); y — из EDL caption_y
SUB_MARGIN = 130                  # поля слева/справа, px
SUB_MAX_WORDS = 3
SUB_MAX_CHARS = 18
SUB_PAUSE = 0.30                  # пауза, после которой начинается новая группа
SUB_TAIL = 0.15                   # группа живёт +0.15 с после последнего слова
YELLOW = "&H004DE1FF&"            # #FFE14D в ASS (AABBGGRR)
WHITE = "&H00FFFFFF&"

TITLE_SIZE = 52
TITLE_SIZE_MIN = 40               # ниже не уменьшаем — переносим на 2 строки
TITLE_SIZE_STEP = 2
TITLE_MAX_W = 1000                # ширина плашки, px
TITLE_LINE_SPACING = 1.15         # межстрочный при переносе на 2 строки
TITLE_RADIUS = 22
TITLE_PAD_X = 28
TITLE_PAD_Y = 16
TITLE_TOP = 636                   # верх плашки

# ---------------------------------------------------------------- константы звука

MUTE_GUARD = 0.002                # с: запас окна глушения (гранулярность volume=enable)
MUTE_FRAME = 64                   # asetnsamples: кадр аудио для точного enable
AFADE = 0.012                     # с, фейд на краях сегмента
BOOM_DB = -9.0                    # дБ относительно полной шкалы
BOOM_DUR = 0.45
BOOM_F0, BOOM_F1 = 110.0, 42.0
BOOM_CLICK = 0.008
LOUD_I, LOUD_TP, LOUD_LRA = -14.0, -1.0, 11.0

# ---------------------------------------------------------------- кодирование

V_ENC = ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-b:v", "0",
         "-profile:v", "high", "-pix_fmt", "yuv420p", "-r", "60",
         "-fps_mode", "cfr", "-video_track_timescale", "60000"]
A_ENC = ["-c:a", "aac", "-b:a", "192k", "-ar", str(SR)]
# Промежуточные — h264 + PCM в MOV: у PCM нет задержки кодера, поэтому
# concat-демуксер режет стыки ровно по длительностям (у AAC-in-MP4 на каждом
# стыке набегало ~43 мс: priming-кадры + edit list).
A_ENC_TMP = ["-c:a", "pcm_s16le", "-ar", str(SR), "-ac", "2"]


def log(msg: str = "") -> None:
    print(msg, flush=True)


def run(cmd, cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace", cwd=cwd)


def ffprobe_duration(path: str) -> float:
    # у ffprobe нет -nostdin (это опция только ffmpeg)
    r = run([FFPROBE, "-hide_banner", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path])
    try:
        return float(r.stdout.strip())
    except ValueError:
        return -1.0


def fail_tail(r: subprocess.CompletedProcess, n: int = 6) -> str:
    tail = "\n".join((r.stderr or "").strip().splitlines()[-n:])
    return f"ffmpeg rc={r.returncode}\n{tail}"


# ================================================================ цензура
# Логика цензуры скопирована из censor.py заимствованного модуля (строки 89-103).
# Adapted from Reelsi (https://github.com/mxmlab/reelsi), AGPL-3.0, reelsi/core/censor.py
_norm_re = re.compile(r"[^\w]+", re.UNICODE)


def _norm_word(s) -> str:
    """Нормализация для сравнения со стемами: нижний регистр, ё→е, только \\w.

    Whisper пишет «уёбок», а стем в списке — «уеб»: без приведения ё→е (и в
    слове, и в стемах) мат не находится и звук не глушится. Звёздочку censor()
    ставит в ИСХОДНОЕ слово — там регистр и ё сохраняются.
    """
    return _norm_re.sub("", (s or "").lower().replace("ё", "е"))


def _parse_stems(lines) -> list:
    out = []
    for ln in lines:
        s = ln.split("#", 1)[0].strip().lower().replace("ё", "е")
        if s:
            out.append(s)
    return out


def load_stems() -> tuple:
    """(bad, ok). bad — ТОЛЬКО стемы раздела после строки '# --- мат'.

    Пути словарей — из конфига (`badwords_path` / `okwords_path`); нет файла —
    понятная ошибка со ссылкой на раздел README «Profanity dictionaries».
    """
    hint = ("словари мата в репозиторий не входят: возьмите готовые\n"
            "https://github.com/mxmlab/reelsi/blob/main/data/badwords.txt и okwords.txt\n"
            "или сгенерируйте свои — см. README, раздел «Profanity dictionaries»")
    for key, what in (("badwords_path", "файл плохих слов (badwords)"),
                      ("okwords_path", "файл исключений (okwords)")):
        if not os.path.isfile(CONFIG.path(key)):
            raise SystemExit("нет %s: %s (ключ %r в %s)\n%s"
                             % (what, CONFIG.path(key), key, CONFIG.path, hint))
    bad_lines, ok_lines = [], []
    with open(BADWORDS_TXT, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    cut = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("# --- мат"):
            cut = i
    if cut is not None:
        bad_lines = lines[cut + 1:]
    with open(OKWORDS_TXT, encoding="utf-8") as fh:
        ok_lines = fh.read().splitlines()
    return _parse_stems(bad_lines), _parse_stems(ok_lines)


def is_bad(word, bad, ok) -> bool:
    """Слово плохое, если содержит плохой стем и не содержит ok-стема."""
    low = _norm_word(word)
    if not low:
        return False
    if any(o in low for o in ok):
        return False
    return any(b in low for b in bad)


def censor(word: str, bad, ok) -> str:
    """Звёздочка на среднюю букву (регистр сохраняется)."""
    if is_bad(word, bad, ok):
        m = max(1, len(word) // 2)
        return word[:m] + "*" + word[m + 1:]
    return word


def mute_windows_of(word: dict, bad, ok) -> list:
    """Окна глушения в секундах ИСХОДНИКА для одного слова.

    Слово глушится, если плохое ЛИБО `w` (текст Whisper), ЛИБО `giga` (исходное
    слово GigaAM, поле из fix_words.py): Whisper «чистит» мат («бля» -> «для»),
    и по одному `w` цензура его не находит.

    Окно — по центру средней буквы (центр прежнего окна), длина
    max(window_frac·d, min_window_sec), обрезано в границы слова [s, e];
    целое слово (совпадение по всему слову) — плюс word_pad_sec с обеих сторон.
    Числа правила — из конфига, раздел `censor`; дефолты: 0.40 / 0.15 / 0.03.
    """
    rule = CONFIG.censor()
    window_frac = float(rule["window_frac"])
    min_window = float(rule["min_window_sec"])
    word_pad = float(rule["word_pad_sec"])
    bad_texts = [t for t in (word.get("w"), word.get("giga"))
                 if t and is_bad(t, bad, ok)]
    if not bad_texts:
        return []
    s, e = float(word["start"]), float(word["end"])
    d = e - s
    if any("пидор" in _norm_word(t) for t in bad_texts):
        return [(s - word_pad, e + word_pad)]   # такие слова глушим целиком
    n = len(bad_texts[0])                    # длина плохого слова (w, если плохое w)
    if n <= 0:
        return []
    m = max(1, n // 2)
    c = s + d * (m + 0.5) / n                # центр прежнего окна
    half = max(window_frac * d, min_window) / 2.0
    ws, we = max(s, c - half), min(e, c + half)
    if we - ws <= 1e-6:
        return []
    return [(ws, we)]


# ================================================================ монтажный лист


def source_video(src: str) -> str:
    return find_source(src)


def load_words(src: str) -> list:
    path = os.path.join(ANALYSIS_DIR, src, "words.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def sidecar_words(edl: dict, edl_path: str):
    """`<name>.words.json` рядом с EDL (результат fix_words.py), если он есть.

    Ищем по имени файла EDL, затем по edl["name"] — fix_words.py кладёт файл
    рядом с EDL. -> (слова, путь) или None.
    """
    d = os.path.dirname(os.path.abspath(edl_path))
    stems = [os.path.splitext(os.path.basename(edl_path))[0]]
    if edl.get("name") and edl["name"] not in stems:
        stems.append(edl["name"])
    for stem in stems:
        path = os.path.join(d, stem + ".words.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                return json.load(fh), path
    return None


def load_words_for(edl: dict, edl_path: str) -> tuple:
    """(слова, описание источника для лога).

    Рядом с EDL лежит `<name>.words.json` (fix_words.py: текст Whisper, время
    GigaAM) — берём слова ИЗ НЕГО, и для субтитров, и для цензуры. Нет файла —
    прежнее поведение: `<analysis_dir>\\<src>\\words.json` (GigaAM).
    """
    sc = sidecar_words(edl, edl_path)
    if sc is not None:
        words, path = sc
        return words, f"{path} (fix_words: слова Whisper, время GigaAM)"
    path = os.path.join(ANALYSIS_DIR, edl["src"], "words.json")
    return load_words(edl["src"]), f"{path} (GigaAM)"


HF_OVERLAY = os.path.join(TT_DIR, "hf_overlay.py")


def run_hf_overlay(edl_path: str, plan: Plan) -> str:
    """Часть C: слой субтитров/заголовка HyperFrames вместо ASS и PNG.

    Отдельным процессом: hf_overlay.py сам решает, рендерить overlay.mov или
    взять готовый (новее EDL/words.json), и печатает свою таблицу проверок
    (код != 0 — слой не собрался).
    """
    t0 = time.time()
    r = subprocess.run([sys.executable, HF_OVERLAY, os.path.abspath(edl_path)],
                       cwd=TT_DIR, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    for line in (r.stdout or "").splitlines():
        log("  | " + line)
    for line in (r.stderr or "").strip().splitlines()[-4:]:
        log("  | " + line)
    mov = os.path.join(TT_DIR, "hf", plan.name, "overlay.mov")
    if r.returncode != 0 or not os.path.exists(mov):
        raise RuntimeError(f"hf_overlay не собрал слой (код {r.returncode}): {mov}")
    log(f"  слой HyperFrames: {mov}  ({time.time() - t0:.1f} с)")
    return mov


def hf_stats(edl_path: str, plan: Plan, bad, ok, mov: str) -> dict:
    """Сводка по слою (группы/слова) для строки проверок render_tt."""
    import hf_overlay as hfo                              # noqa: PLC0415
    sched = hfo.build_schedule(plan, plan.edl, bad, ok)
    dur = ffprobe_duration(mov)
    return {"path": mov, "words": len(sched["rail_words"]),
            "groups": len(sched["groups"]), "dur": dur,
            "ok": dur > 0 and abs(dur - plan.total) <= 1.5 / 60.0}


def parse_game_view(gv) -> tuple:
    """EDL game_view {"crop":[x,y,w,h]} -> (x,y,w,h) в пикселях кадра исходника."""
    if not gv:
        return None
    crop = gv.get("crop") if isinstance(gv, dict) else None
    if not crop or len(crop) != 4:
        raise ValueError('game_view: ожидается {"crop":[x,y,w,h]}')
    x, y, w, h = (int(round(float(v))) for v in crop)
    if w <= 0 or h <= 0:
        raise ValueError(f"game_view: w и h должны быть > 0 (получено {w}x{h})")
    if x < 0 or y < 0 or x + w > SRC_W or y + h > SRC_H:
        raise ValueError(f"game_view: {x},{y},{w},{h} выходит за кадр "
                         f"{SRC_W}x{SRC_H}")
    return (x, y, w, h)


def parse_game_broll(gb) -> float:
    """EDL game_broll {"t": <секунда исходника>} -> float (секунда) или None.

    Заданное t — секунда ИСХОДНИКА, с которой берётся видеоряд игры для
    подложки (кадр t + локальное время сегмента). Панель игры и размытый фон
    тогда идут из второго входа ffmpeg, вебка — из первого.
    """
    if gb is None:
        return None
    t = gb.get("t") if isinstance(gb, dict) else None
    if t is None:
        raise ValueError('game_broll: ожидается {"t": <секунда исходника>}')
    return float(t)


class Plan:
    """Расписание выходного ролика: сегменты, слова, панчи, окна глушения."""

    def __init__(self, edl: dict, words: list, bad, ok):
        self.edl = edl
        self.name = edl["name"]
        self.src = edl["src"]
        self.title = edl.get("title")
        self.caption_y = int(edl.get("caption_y", SUB_POS[1]))
        self.cam_alt = bool(edl.get("cam_alt"))     # чередование крупности вебки
        self.game_view = parse_game_view(edl.get("game_view"))
        self.game_broll = parse_game_broll(edl.get("game_broll"))
        self.segments = []
        t = 0.0
        for i, seg in enumerate(edl.get("segments") or []):
            a, b = float(seg["a"]), float(seg["b"])
            speed = float(seg.get("speed", 1.0) or 1.0)
            if b <= a:
                raise ValueError(f"сегмент {i}: b <= a ({a}..{b})")
            if not (1.0 <= speed <= 2.0):
                raise ValueError(f"сегмент {i}: speed={speed} вне 1.0..2.0")
            if self.game_broll is not None and abs(speed - 1.0) > 1e-9:
                raise ValueError(f"сегмент {i}: game_broll несовместим со "
                                 f"speed={speed} (нужен speed=1)")
            dur = (b - a) / speed
            self.segments.append({"i": i, "a": a, "b": b, "speed": speed,
                                  "dur": dur, "out": t})
            t += dur
        if not self.segments:
            raise ValueError("в EDL нет сегментов")
        self.total = t

        caption_off = [(float(x[0]), float(x[1])) for x in (edl.get("caption_off") or [])]
        extra_mute = [(float(x[0]), float(x[1])) for x in (edl.get("extra_mute") or [])]

        # --- слова: только целиком внутри сегмента и не в caption_off
        self.words = []          # показанные слова (выходное время)
        self.groups = []         # группы субтитров
        for seg in self.segments:
            a, b, sp, out0 = seg["a"], seg["b"], seg["speed"], seg["out"]
            inside = [w for w in words
                      if float(w["start"]) >= a - 1e-9 and float(w["end"]) <= b + 1e-9]
            inside = [w for w in inside
                      if not any(float(w["start"]) < co1 and float(w["end"]) > co0
                                 for co0, co1 in caption_off)]
            inside.sort(key=lambda w: float(w["start"]))
            shown = []
            for w in inside:
                shown.append({"w": w["w"],
                              "s": out0 + (float(w["start"]) - a) / sp,
                              "e": out0 + (float(w["end"]) - a) / sp,
                              "seg": seg["i"],
                              # время ИСХОДНИКА — для EDL-полей emph/drop (hf_overlay)
                              "src_s": float(w["start"]),
                              "src_e": float(w["end"])})
            self.words.extend(shown)

            # --- группы: <=3 слов, <=18 символов, пауза >0.30 с рвёт группу
            groups, cur = [], []
            for w in shown:
                if cur:
                    gap = w["s"] - cur[-1]["e"]
                    chars = sum(len(x["w"]) for x in cur) + len(cur) + len(w["w"])
                    if len(cur) >= SUB_MAX_WORDS or chars > SUB_MAX_CHARS or gap > SUB_PAUSE:
                        groups.append(cur)
                        cur = []
                cur.append(w)
            if cur:
                groups.append(cur)
            for gi, g in enumerate(groups):
                # конец группы: +0.15 с после последнего слова, но не заходя на следующую
                end = g[-1]["e"] + SUB_TAIL
                if gi + 1 < len(groups):
                    end = min(end, groups[gi + 1][0]["s"])
                end = min(end, out0 + seg["dur"])
                g_end = max(end, g[-1]["e"])
                self.groups.append({"words": g, "start": g[0]["s"], "end": g_end})

        # --- punch: время ИСХОДНИКА -> срабатывает в КАЖДОМ сегменте, где t in [a,b)
        self.punches = []
        for p in (edl.get("punch") or []):
            pt = float(p["t"])
            focus = p.get("focus", "cam")
            if focus not in ("cam", "game"):
                raise ValueError(f"punch focus={focus!r} (ожидается cam|game)")
            for seg in self.segments:
                if seg["a"] <= pt < seg["b"]:
                    self.punches.append({
                        "t": seg["out"] + (pt - seg["a"]) / seg["speed"],
                        "dur": float(p.get("dur", 0.8)) / seg["speed"],
                        "scale": float(p.get("scale", 1.2)),
                        "focus": focus,
                        "boom": bool(p.get("boom", False)),
                    })
        self.punches.sort(key=lambda p: p["t"])

        # --- окна глушения: локальное время сегмента + выходное время (для проверки)
        self.mutes = []          # [{'seg': i, 'local': (s,e), 'out': (s,e)}]
        for seg in self.segments:
            a, b, sp, out0 = seg["a"], seg["b"], seg["speed"], seg["out"]
            wins = []
            for w in words:
                ws, we = float(w["start"]), float(w["end"])
                if we <= a or ws >= b:
                    continue
                wins.extend(mute_windows_of(w, bad, ok))
            wins.extend(extra_mute)
            for ws, we in wins:
                s, e = max(ws, a), min(we, b)
                if e - s <= 1e-6:
                    continue
                self.mutes.append({
                    "seg": seg["i"],
                    "local": (s - a, e - a),
                    "local_out": ((s - a) / sp, (e - a) / sp),
                    "out": (out0 + (s - a) / sp, out0 + (e - a) / sp),
                })
        self.mutes.sort(key=lambda m: m["out"][0])

    # ---------------------------------------------------------- проверки п.4

    def cam_alt_seg(self, i: int) -> bool:
        """Сегмент с нечётным порядковым номером (1, 3, 5…) при `cam_alt`."""
        return self.cam_alt and (i % 2 == 0)

    def expected_duration(self) -> float:
        return sum(s["dur"] for s in self.segments)


# ================================================================ заголовок/субтитры


def _title_layout(text: str) -> tuple:
    """(размер шрифта, строки) по правилу: ширина плашки <= TITLE_MAX_W.

    Сначала 52 px; если плашка шире — шаг 2 px вниз до 40 px; если и при 40 px
    не влезает — перенос на 2 строки (по пробелу ближе к середине) при 52 px.
    """
    tmp = ImageDraw.Draw(Image.new("RGBA", (8, 8)))

    def fits(size: int) -> bool:
        font = ImageFont.truetype(FONT, size)
        return tmp.textlength(text, font=font) + 2 * TITLE_PAD_X <= TITLE_MAX_W

    for size in range(TITLE_SIZE, TITLE_SIZE_MIN - 1, -TITLE_SIZE_STEP):
        if fits(size):
            return size, [text]

    mid = len(text) / 2.0
    spaces = [i for i, ch in enumerate(text) if ch == " "]
    if not spaces:
        return TITLE_SIZE_MIN, [text]
    cut = min(spaces, key=lambda i: abs(i - mid))
    return TITLE_SIZE, [text[:cut], text[cut + 1:]]


def make_title_png(text: str, path: str) -> dict:
    """Плашка с заголовком (верх y=TITLE_TOP, центр по x). -> описание плашки.

    Вертикальная раскладка — по метрикам, а не по чернилам: строка рисуется от
    базовой линии (anchor="ls"), высота строки — cap_h, высота заглавной "Н"
    этого шрифта. Блок — от верха заглавных первой строки до базовой линии
    последней, поэтому отступы TITLE_PAD_Y сверху и снизу равны.
    """
    size, lines = _title_layout(text)
    font = ImageFont.truetype(FONT, size)
    tmp = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    boxes = [tmp.textbbox((0, 0), ln, font=font) for ln in lines]
    widths = [b[2] - b[0] for b in boxes]
    cap_h = -tmp.textbbox((0, 0), "Н", font=font, anchor="ls")[1]
    step = size * TITLE_LINE_SPACING
    bases = [round(cap_h + i * step) for i in range(len(lines))]
    block_h = bases[-1]
    text_w = max(widths)
    pw = int(round(text_w)) + 2 * TITLE_PAD_X
    ph = block_h + 2 * TITLE_PAD_Y
    x0 = (OUT_W - pw) // 2
    y0 = TITLE_TOP
    img = Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((x0, y0, x0 + pw - 1, y0 + ph - 1), radius=TITLE_RADIUS,
                        fill=(255, 255, 255, 255))
    for i, ln in enumerate(lines):
        bx = boxes[i]
        tx = x0 + TITLE_PAD_X + (text_w - widths[i]) / 2 - bx[0]
        ty = y0 + TITLE_PAD_Y + bases[i]
        d.text((tx, ty), ln, font=font, anchor="ls", fill=(0, 0, 0, 255))
    img.save(path)
    return {"path": path, "size": size, "lines": lines, "pw": pw, "ph": ph}


def _text_w(text: str, font) -> float:
    d = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    return d.textlength(text, font=font)


def wrap_words(words: list) -> list:
    """Разбить слова группы на строки так, чтобы каждая влезала в поля."""
    font = ImageFont.truetype(FONT, SUB_SIZE)
    limit = OUT_W - 2 * SUB_MARGIN
    lines, cur = [], []
    for w in words:
        trial = " ".join([x["w"] for x in cur] + [w["w"]])
        if cur and _text_w(trial, font) > limit:
            lines.append(cur)
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(cur)
    return lines


def ass_time(t: float) -> str:
    t = max(0.0, t)
    h = int(t // 3600)
    m = int((t - h * 3600) // 60)
    s = t - h * 3600 - m * 60
    return f"{h}:{m:02d}:{s:05.2f}"


def build_ass(plan: Plan, path: str, bad, ok) -> tuple:
    """ASS: по одному Dialogue на слово. -> (путь, число событий)."""
    font = ImageFont.truetype(FONT, SUB_SIZE)
    head = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n"
        "WrapStyle: 2\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: TV.709\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour,"
        " BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle,"
        " BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: TT,Montserrat ExtraBold,{SUB_SIZE},{WHITE},&H000000FF,&H00000000,"
        f"&H00000000,0,0,0,0,100,100,0,0,1,{SUB_OUTLINE},0,5,{SUB_MARGIN},{SUB_MARGIN},0,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    pos = f"{{\\an5\\pos({SUB_POS[0]},{plan.caption_y})}}"
    events, n_ev = [], 0

    for g in plan.groups:
        ws = g["words"]
        lines = wrap_words(ws)
        # стилизованный текст: активное слово жёлтое и «прыгает»
        for i, w in enumerate(ws):
            active = (f"{{\\c{YELLOW}\\fscx118\\fscy118"
                      f"\\t(0,90,\\fscx100\\fscy100)}}"
                      f"{censor(w['w'], bad, ok).upper()}"
                      f"{{\\c{WHITE}\\fscx100\\fscy100}}")
            body = []
            for line in lines:
                parts = []
                for x in line:
                    if x is w:
                        parts.append(active)
                    else:
                        parts.append(censor(x["w"], bad, ok).upper())
                body.append(" ".join(parts))
            text = pos + "\\N".join(body)
            start = w["s"]
            end = ws[i + 1]["s"] if i + 1 < len(ws) else g["end"]
            if end <= start:
                end = start + 0.05
            events.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},TT,,0,0,0,,{text}")
            n_ev += 1

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(head + "\n".join(events) + "\n")
    return path, n_ev


# ================================================================ бум


def make_boom(path: str) -> str:
    """Синус 110->42 Гц за 0.45 с, атака 5 мс / спад exp(-t/0.12) + щелчок 8 мс."""
    n = int(SR * BOOM_DUR)
    t = np.arange(n) / SR
    freq = BOOM_F0 * (BOOM_F1 / BOOM_F0) ** (t / BOOM_DUR)
    phase = 2.0 * np.pi * np.cumsum(freq) / SR
    sig = np.sin(phase) * np.where(t < 0.005, t / 0.005, np.exp(-(t - 0.005) / 0.12))
    rng = np.random.RandomState(0)
    nc = int(SR * BOOM_CLICK)
    tc = np.arange(nc) / SR
    click = rng.uniform(-1.0, 1.0, nc) * np.exp(-tc / 0.0015)
    sig[:nc] += click
    sig /= np.max(np.abs(sig)) or 1.0
    sf.write(path, np.stack([sig, sig], axis=1), SR, subtype="PCM_16")
    return path


# ================================================================ фильтры


def bg_chain(gx0: int, vin: str = "0:v") -> str:
    """Размытый фон из игрового прямоугольника, затемнённый до BG_BRIGHTNESS.

    vin — откуда берётся игровой прямоугольник: имя входа ("0:v") либо метка
    уже готового потока ("gb" — выход split второго входа при game_broll,
    там подложка игры читается из [1:v]).

    romax/gomax/bomax — ВЫХОДНАЯ белая точка colorlevels: кадр домножается на
    0.45 (затемнение до 45 %). rimax/gimax/bimax (как в rv.build_filter) —
    входная: она растягивает 0..0.45 в 0..1, то есть ОСВЕТЛЯЕТ и уводит в
    белое всё, что выше 0.45.
    """
    b = rv.BG_BRIGHTNESS
    return (
        f"[{vin}]crop={rv.GAME_W}:{rv.GAME_H}:{gx0}:0,"
        f"scale={rv.BG_SMALL[0]}:{rv.BG_SMALL[1]},"
        f"gblur=sigma={rv.BG_BLUR_SIGMA},"
        f"scale={OUT_W}:{OUT_H},"
        f"colorlevels=romax={b}:gomax={b}:bomax={b}[bg]"
    )


def layout_filter(cam, gx0, game_view=None, broll: bool = False) -> str:
    """Раскладка сегмента. Без game_view — как rv.build_filter, с ним — своя игра.

    broll=True (EDL game_broll): подложка — размытый фон и игровая панель —
    берётся из ВТОРОГО входа ([1:v], тот же исходник с другого места стрима),
    вебка — из первого ([0:v]): у второго входа СВОЁ время, и вебка, срезанная
    из него, показывала бы кадр подложки, а не сегмента. broll=False — фильтр
    прежний, байт-в-байт.

    overlay.png (полоса + плашка twitch) здесь НЕ накладывается: он кладётся
    один раз в финальном проходе, после всех зумов (punch/cam_alt) — иначе зум
    финального прохода увеличивал бы уже впечённую в сегмент копию плашки.
    """
    cw, ch, cx, cy = cam
    # ветка вебки: при broll — прямо с первого входа (split не нужен: с [0:v]
    # снимается ровно один поток), иначе — как было, с [g1] из split
    cam_src = "[0:v]" if broll else "[g1]"
    # split первого входа нужен только без broll: там все три ветки (фон, вебка,
    # панель игры) берутся с [0:v]
    if broll:
        head, bg_src, game_src = "[1:v]split=2[g2][gb]", bg_chain(gx0, "gb"), "[g2]"
    else:
        head, bg_src, game_src = "[0:v]split=3[g1][g2][g3]", bg_chain(gx0, "0:v"), "[g2]"
    # шапка графа до ветки вебки: без broll — split первого входа и фон с [0:v],
    # при broll — split второго входа ([g2] панель игры, [gb] фон)
    parts0 = [x for x in (head, bg_src) if x]

    if not game_view:
        # копия rv.build_filter; отличается фоном (bg_chain с romax, а не
        # rimax, который фон не затемняет, а осветляет) и отсутствием [1:v]
        parts = parts0 + [
            f"{cam_src}crop={cw}:{ch}:{cx}:{cy},"
            f"scale={rv.CAM_OUT[0]}:{rv.CAM_OUT[1]}[cam]",
            f"{game_src}crop={rv.GAME_W}:{rv.GAME_H}:{gx0}:0,"
            f"scale={rv.GAME_OUT[0]}:{rv.GAME_OUT[1]}[game]",
            "[bg][cam]overlay=0:0:format=auto:shortest=0[o1]",
            f"[o1][game]overlay=0:{rv.GAME_Y}:format=auto:shortest=0,setsar=1,"
            f"format=yuv420p[v]",
        ]
        if not broll:                     # [g3] иначе остался бы без потребителя
            parts.append("[g3]nullsink")
        return ";".join(parts)

    # игра берётся из указанного прямоугольника исходника: по ширине -> 1080,
    # высота = h*1080/w, центр — по вертикали игровой панели (612..1826).
    x, y, w, h = game_view
    gw = OUT_W
    gh = 2 * int(round(h * OUT_W / (2.0 * w)))            # чётная, пропорции те же
    gy = int(math.floor(rv.GAME_Y + rv.GAME_OUT[1] / 2.0 - gh / 2.0 + 0.5))

    parts = parts0 + [
        f"{cam_src}crop={cw}:{ch}:{cx}:{cy},"
        f"scale={rv.CAM_OUT[0]}:{rv.CAM_OUT[1]}[cam]",
        f"{game_src}crop={w}:{h}:{x}:{y},scale={gw}:{gh},setsar=1[game]",
        "[bg][cam]overlay=0:0:format=auto:shortest=0[o1]",
        f"[o1][game]overlay=0:{gy}:format=auto:shortest=0,setsar=1,"
        f"format=yuv420p[v]",
    ]
    if not broll:
        parts.append("[g3]nullsink")
    return ";".join(parts)


def panel_zoom_chain(panel: tuple, events: list, base_spans: list = None) -> str:
    """crop панели -> scale Z(t) -> crop центра обратно в WxH.

    base_spans — интервалы выходного времени с постоянным зумом CAM_ALT_SCALE
    (cam_alt: нечётные сегменты); punch в них считается уже от неё, итог = 1.07·Z.
    """
    w, h, y = panel
    if events:
        terms = [f"{max(p['scale'] - 1.0, 0.0):.6f}*"
                 f"min(max(0,min((t-{p['t']:.4f})/{PUNCH_ATTACK},"
                 f"({p['t'] + p['dur']:.4f}-t)/{PUNCH_RELEASE})),1)"
                 for p in events]
        z = "1+" + (terms[0] if len(terms) == 1 else _nested_max(terms))
    else:
        z = "1"
    if base_spans:
        ind = "+".join(f"gte(t,{s:.4f})*lt(t,{e:.4f})" for s, e in base_spans)
        z = f"(1+{CAM_ALT_SCALE - 1.0:.6f}*({ind}))*({z})"
    return (f"crop={w}:{h}:0:{y},"
            f"scale=w='{w}*({z})':h='{h}*({z})':eval=frame,"
            f"crop={w}:{h}:'({w}*({z})-{w})/2':'({h}*({z})-{h})/2'")


def _nested_max(terms: list) -> str:
    out = terms[-1]
    for t in reversed(terms[:-1]):
        out = f"max({t},{out})"
    return out


def _ff_path(path: str) -> str:
    """Путь для filter_complex: ':' -> '\\:' (двоеточие ломает опции фильтра)."""
    return path.replace("\\", "/").replace(":", "\\:")


def audio_chain(plan: Plan, seg: dict, loudnorm: str = "") -> str:
    """[0:a] -> глушение окон -> скорость -> фейды 12 мс -> (loudnorm).

    Окна глушения ставятся дважды: до atempo (как в TASK.md) и, при speed != 1,
    ещё раз после atempo по пересчитанному выходному времени — WSOLA-переходы
    atempo размазывают края тишины на ~30 мс, и без второго глушения окно
    проверки п.4 перестаёт быть тихим.
    """
    steps = []
    mutes = [m for m in plan.mutes if m["seg"] == seg["i"]]
    lim = seg["b"] - seg["a"]

    def gates(key, hi):
        for m in mutes:
            _s, _e = m[key]
            s = max(0.0, _s - MUTE_GUARD)
            e = min(hi, _e + MUTE_GUARD)
            if e > s:
                steps.append(f"volume=0:enable='between(t,{s:.4f},{e:.4f})'")

    if mutes:
        steps.append(f"asetnsamples=n={MUTE_FRAME}:p=0")
        gates("local", lim)
    if abs(seg["speed"] - 1.0) > 1e-9:
        steps.append(f"atempo={seg['speed']:.6f}")
        if mutes:
            steps.append(f"asetnsamples=n={MUTE_FRAME}:p=0")
            gates("local_out", seg["dur"])
    steps.append(f"afade=t=in:st=0:d={AFADE}")
    steps.append(f"afade=t=out:st={max(seg['dur'] - AFADE, 0.0):.4f}:d={AFADE}")
    if loudnorm:
        steps.append(loudnorm)
    return "[0:a]" + ",".join(steps) + "[aout]"


def render_segment(plan: Plan, seg: dict, src: str, cam, gx0, dst: str) -> tuple:
    broll = plan.game_broll is not None
    fc = layout_filter(cam, gx0, plan.game_view, broll=broll)
    if abs(seg["speed"] - 1.0) > 1e-9:
        fc += f";[v]setpts=PTS/{seg['speed']:.6f}[vs]"
    else:
        fc += ";[v]null[vs]"
    a = audio_chain(plan, seg)
    if broll:
        # звук всегда из сегмента (первый вход), у второго входа он только для
        # формы: нумерация потоков остаётся 1:v / 1:a, а аудио уходит в nullsink
        a += ";[1:a]anullsink"
    fc += ";" + a

    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-y",
           "-ss", f"{seg['a']:.6f}", "-to", f"{seg['b']:.6f}", "-i", src]
    if broll:
        # подложка игры: тот же исходник, но с game_broll.t + начало сегмента
        # в выходном времени; панель/фон берутся из этого входа (layout_filter)
        cmd += ["-ss", f"{plan.game_broll + seg['out']:.6f}",
                "-t", f"{seg['dur']:.6f}", "-i", src]
    cmd += ["-filter_complex", fc,
            "-map", "[vs]", "-map", "[aout]",
            *V_ENC, "-cq", "14", *A_ENC_TMP,
            "-t", f"{seg['dur']:.6f}",
            dst]
    t0 = time.time()
    r = run(cmd)
    dt = time.time() - t0
    if r.returncode != 0 or not os.path.exists(dst):
        return False, fail_tail(r) + f"\nкоманда: {' '.join(cmd)}"
    return True, f"{dt:.1f}s"


def alt_spans(plan: Plan) -> list:
    """Интервалы выходного времени с постоянным зумом вебки (cam_alt)."""
    if not plan.cam_alt:
        return []
    return [(s["out"], s["out"] + s["dur"]) for s in plan.segments
            if plan.cam_alt_seg(s["i"])]


def zoom_active(plan: Plan) -> bool:
    """Есть ли в финальном проходе зум панели: punch (cam|game) или cam_alt."""
    return bool(plan.punches) or bool(alt_spans(plan))


def build_final_filter(plan: Plan, ass_name: str, has_title: bool, title_idx: int,
                       booms: list, boom_idx, loudnorm: str, hf_idx: int = None,
                       ovl_idx: int = None) -> str:
    focus_events = {"cam": [p for p in plan.punches if p["focus"] == "cam"],
                    "game": [p for p in plan.punches if p["focus"] == "game"]}
    # cam_alt — постоянный зум панели вебки в нечётных сегментах (1.07 от центра,
    # как punch, но на весь сегмент); punch с focus cam считается уже от неё
    alt = alt_spans(plan)
    active = [f for f in ("cam", "game")
              if focus_events[f] or (f == "cam" and alt)]
    parts = []
    if active:
        labels = ["[base]"] + [f"[p_{f}]" for f in active]
        parts.append(f"[0:v]split={len(labels)}" + "".join(labels))
        cur = "base"
    else:
        cur = "0:v"
    panel_of = {"cam": CAM_PANEL, "game": GAME_PANEL}
    for k, f in enumerate(active):
        w, h, y = panel_of[f]
        spans = alt if f == "cam" else None
        parts.append(f"[p_{f}]{panel_zoom_chain(panel_of[f], focus_events[f], spans)}"
                     f"[z_{f}]")
        parts.append(f"[{cur}][z_{f}]overlay=0:{y}:format=auto:shortest=0[o{k}]")
        cur = f"o{k}"
    if ovl_idx is not None:
        # overlay.png (полоса + плашка канала, надпись — --overlay-text в rv) кладётся здесь ОДИН
        # раз — после зумов панелей (punch/cam_alt) и до заголовка/HF-слоя.
        # В сегментах он не впекается (layout_filter): иначе зум финального
        # прохода увеличивал бы его копию в кадре — плашка двоилась
        parts.append(f"[{ovl_idx}:v]format=rgba,scale={OUT_W}:{OUT_H}[ovr]")
        parts.append(f"[{cur}][ovr]overlay=0:0:format=auto:shortest=0[oov]")
        cur = "oov"
    if has_title:
        parts.append(f"[{cur}][{title_idx}:v]overlay=0:0:format=auto:shortest=0[ot]")
        cur = "ot"
    if hf_idx is not None:
        # style:"hf" — вместо ASS-субтитров и PNG-заголовка прозрачный слой
        # HyperFrames поверх всего (после punch/зума), затем кодирование
        parts.append(f"[{cur}][{hf_idx}:v]overlay=0:0:format=auto:shortest=0[oh]")
        parts.append("[oh]setsar=1,format=yuv420p[vout]")
    else:
        parts.append(f"[{cur}]subtitles={ass_name}:"
                     f"fontsdir='{_ff_path(FONTS_DIR)}',setsar=1,format=yuv420p[vout]")

    # --- звук
    # [aout]  — то, что идёт в ролик (с бумом);
    # [acheck] — тот же звук после того же loudnorm, но БЕЗ бума: по нему
    #           проверяются окна цензуры, иначе бум даёт ложный провал.
    a = ["[0:a]aformat=sample_rates=48000:channel_layouts=stereo[main0]",
         "[main0]asplit=2[main][mainchk]"]
    if booms:
        if len(booms) == 1:
            a.append(f"[{boom_idx}:a]adelay=delays={int(round(booms[0] * 1000))}:all=1,"
                     f"volume={BOOM_DB}dB[b0d]")
            srcs = ["[main]", "[b0d]"]
        else:
            a.append(f"[{boom_idx}:a]asplit={len(booms)}" +
                     "".join(f"[b{i}]" for i in range(len(booms))))
            srcs = ["[main]"]
            for i, b in enumerate(booms):
                ms = int(round(b * 1000))
                a.append(f"[b{i}]adelay=delays={ms}:all=1,volume={BOOM_DB}dB[b{i}d]")
                srcs.append(f"[b{i}d]")
        a.append("".join(srcs) + f"amix=inputs={len(srcs)}:duration=first:"
                                 f"normalize=0[a_mix]")
        amix = "a_mix"
    else:
        amix = "main"
    a.append(f"[{amix}]aformat=sample_rates=48000:channel_layouts=stereo{loudnorm}[aout]")
    a.append(f"[mainchk]aformat=sample_rates=48000:channel_layouts=stereo"
             f"{loudnorm}[acheck]")
    return ";".join(parts + a)


def measure_loudness(plan: Plan, joined: str, boom_idx, booms: list) -> dict:
    """Первый проход loudnorm — только замер (звук ровно тот же, что в финале)."""
    a = ["[0:a]aformat=sample_rates=48000:channel_layouts=stereo[main]"]
    if booms:
        if len(booms) == 1:
            a.append(f"[{boom_idx}:a]adelay=delays={int(round(booms[0] * 1000))}:all=1,"
                     f"volume={BOOM_DB}dB[b0d]")
            srcs = ["[main]", "[b0d]"]
        else:
            a.append(f"[{boom_idx}:a]asplit={len(booms)}" +
                     "".join(f"[b{i}]" for i in range(len(booms))))
            srcs = ["[main]"]
            for i, b in enumerate(booms):
                ms = int(round(b * 1000))
                a.append(f"[b{i}]adelay=delays={ms}:all=1,volume={BOOM_DB}dB[b{i}d]")
                srcs.append(f"[b{i}d]")
        a.append("".join(srcs) + f"amix=inputs={len(srcs)}:duration=first:normalize=0[a_mix]")
        amix = "a_mix"
    else:
        amix = "main"
    a.append(f"[{amix}]loudnorm=I={LOUD_I}:TP={LOUD_TP}:LRA={LOUD_LRA}:print_format=json[aout]")
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-i", joined]
    if boom_idx is not None:
        cmd += ["-i", BOOM_WAV]
    cmd += ["-filter_complex", ";".join(a), "-map", "[aout]", "-f", "null", "-"]
    r = run(cmd)
    if r.returncode != 0:
        raise RuntimeError("замер loudnorm не удался:\n" + fail_tail(r))
    blocks = re.findall(r"\{[^{}]*\}", r.stderr or "")
    for b in reversed(blocks):
        if "input_i" in b:
            return json.loads(b)
    raise RuntimeError("loudnorm не вернул JSON:\n" + (r.stderr or "")[-800:])


# ================================================================ проверка п.4


def probe_video(path: str) -> dict:
    r = run([FFPROBE, "-hide_banner", "-v", "error",
             "-select_streams", "v:0", "-show_entries",
             "stream=width,height,sample_aspect_ratio,display_aspect_ratio,r_frame_rate",
             "-of", "json", path])
    try:
        return json.loads(r.stdout)["streams"][0]
    except Exception:
        return {}


def integrated_lufs(path: str) -> float:
    r = run([FFMPEG, "-hide_banner", "-nostdin", "-i", path, "-af",
             f"ebur128=peak=true:framelog=quiet", "-f", "null", "-"])
    m = re.findall(r"I:\s*(-?\d+(?:\.\d+)?)\s*LUFS", r.stderr or "")
    return float(m[-1]) if m else float("nan")


def audio_to_array(path: str) -> tuple:
    """Раскодировать звук в numpy (для проверки окон глушения — файл без бума)."""
    r = subprocess.run([FFMPEG, "-hide_banner", "-nostdin", "-i", path, "-vn",
                        "-ac", "1", "-ar", str(SR), "-f", "wav", "-"],
                       capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("не смог раскодировать звук:\n" +
                           "\n".join((r.stderr or b"").decode("utf-8", "replace")
                                     .strip().splitlines()[-6:]))
    import io
    x, sr = sf.read(io.BytesIO(r.stdout))
    return x, sr


def window_rms_db(x: np.ndarray, sr: int, s: float, e: float) -> float:
    a, b = max(0, int(round(s * sr))), min(len(x), int(round(e * sr)))
    if b <= a:
        return float("nan")
    seg = x[a:b].astype(np.float64)
    return 20.0 * math.log10(float(np.sqrt((seg ** 2).mean())) + 1e-12)


def verify(plan: Plan, out_path: str, ass_path: str, n_events: int, audio,
           hf: dict = None) -> list:
    rows = []

    def add(name, expected, got, ok):
        rows.append((name, expected, got, bool(ok)))

    # 1. длительность
    exp_dur = plan.expected_duration()
    got_dur = ffprobe_duration(out_path)
    add("длительность, с", f"Σ(b−a)/speed = {exp_dur:.3f} ±0.10",
        f"{got_dur:.3f}", abs(got_dur - exp_dur) <= 0.10)

    # 2. геометрия и fps
    st = probe_video(out_path)
    add("кадр", "1080x1920", f"{st.get('width')}x{st.get('height')}",
        st.get("width") == 1080 and st.get("height") == 1920)
    add("SAR / DAR", "1:1 / 9:16",
        f"{st.get('sample_aspect_ratio')} / {st.get('display_aspect_ratio')}",
        st.get("sample_aspect_ratio") == "1:1" and st.get("display_aspect_ratio") == "9:16")
    add("частота кадров", "60/1", str(st.get("r_frame_rate")),
        st.get("r_frame_rate") == "60/1")

    # 3. громкость
    lufs = integrated_lufs(out_path)
    add("интегральная громкость", "-14 ±1 LUFS", f"{lufs:.2f} LUFS",
        abs(lufs + 14.0) <= 1.0)

    # 4. окна цензуры
    x, sr = audio
    checked = [m for m in plan.mutes if (m["out"][1] - m["out"][0]) >= 0.025]
    skipped = len(plan.mutes) - len(checked)
    if not checked:
        add("окна цензуры (RMS < −55 dBFS)",
            "окон короче 25 мс не проверяем",
            f"окон нет ({skipped} короче 25 мс)", True)
    else:
        worst, worst_m = -999.0, None
        for m in checked:
            db = window_rms_db(x, sr, m["out"][0], m["out"][1])
            if db > worst:
                worst, worst_m = db, m
        add(f"окна цензуры, {len(checked)} шт (RMS < −55 dBFS)",
            "максимум по окнам < −55 dBFS (звук без бума)",
            f"{worst:.1f} dBFS @ {worst_m['out'][0]:.3f}..{worst_m['out'][1]:.3f} с"
            + (f" (+{skipped} короче 25 мс)" if skipped else ""),
            worst < -55.0)

    # 5. субтитры
    if hf:
        add("субтитры — слой HyperFrames", f"{hf['words']} слов в {hf['groups']} группах",
            f"{os.path.basename(hf['path'])}, {hf['dur']:.2f} с "
            f"(ролик {plan.total:.2f} с)",
            hf["ok"])
    else:
        with open(ass_path, encoding="utf-8") as fh:
            n_dial = sum(1 for ln in fh if ln.startswith("Dialogue:"))
        add("события Dialogue = слов", f"{len(plan.words)}",
            f"{n_dial}" + ("" if n_dial == n_events else f" (сгенерировано {n_events})"),
            n_dial == len(plan.words) == n_events)
    return rows


def print_table(title: str, rows: list) -> None:
    w0 = max(len(r[0]) for r in rows)
    w1 = max(len(r[1]) for r in rows)
    w2 = max(len(r[2]) for r in rows)
    log("")
    log(f"  Приёмка: {title}")
    log("  " + "-" * (w0 + w1 + w2 + 12))
    log(f"  {'проверка'.ljust(w0)} | {'ожидалось'.ljust(w1)} | {'получено'.ljust(w2)} | итог")
    log("  " + "-" * (w0 + w1 + w2 + 12))
    for n, e, g, ok in rows:
        log(f"  {n.ljust(w0)} | {e.ljust(w1)} | {g.ljust(w2)} | {'OK' if ok else 'ПРОВАЛ'}")
    log("  " + "-" * (w0 + w1 + w2 + 12))
    log(f"  Итог: {'все проверки пройдены' if all(r[3] for r in rows) else 'ЕСТЬ ПРОВАЛЫ'}")
    log("")


# ================================================================ рендер одного EDL


def render_edl(edl_path: str, sheet: bool = False, keep_temp: bool = False,
               out_dir: str = None) -> bool:
    out_dir = out_dir or OUT_DIR
    t_all = time.time()
    with open(edl_path, encoding="utf-8") as fh:
        edl = json.load(fh)
    name = edl["name"]
    src = source_video(edl["src"])
    layout = edl.get("layout", edl["src"])
    if layout not in rv.LAYOUT:
        raise RuntimeError(f"нет раскладки для исходника '{layout}'")
    if not os.path.exists(src):
        raise RuntimeError(f"нет исходника {src}")

    bad, ok = load_stems()
    words, words_src = load_words_for(edl, edl_path)
    plan = Plan(edl, words, bad, ok)

    temp = os.path.join(TEMP_ROOT, name)
    if os.path.isdir(temp):
        shutil.rmtree(temp, ignore_errors=True)
    os.makedirs(temp, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    log("=" * 78)
    log(f"EDL: {edl_path}")
    log(f"  источник слов: {words_src} — {len(words)} шт")
    log(f"  name={name}  src={edl['src']}  сегментов={len(plan.segments)}  "
        f"длительность={plan.total:.2f} с  слов={len(plan.words)}  "
        f"групп={len(plan.groups)}  punch={len(plan.punches)}  "
        f"окон глушения={len(plan.mutes)}  title={plan.title!r}")

    cam, gx0 = rv.LAYOUT[layout]
    ovl = rv.OVERLAY
    if not os.path.exists(ovl):
        rv.make_overlay(ovl)
    if plan.game_view:
        log(f"  game_view: crop={plan.game_view[2]}x{plan.game_view[3]}"
            f"+{plan.game_view[0]}+{plan.game_view[1]} исходника")
    if plan.game_broll is not None:
        log(f"  game_broll: t={plan.game_broll:.3f} — подложка игры (панель и фон) "
            f"из этого места исходника, вебка и звук из сегментов")

    # --- 1. сегменты (overlay.png здесь не впекается — только в финале)
    seg_files = []
    for seg in plan.segments:
        dst = os.path.join(temp, f"seg_{seg['i']:03d}.mov")
        good, msg = render_segment(plan, seg, src, cam, gx0, dst)
        if not good:
            raise RuntimeError(f"сегмент {seg['i']} не отрендерился:\n{msg}")
        seg_files.append(os.path.basename(dst))
        log(f"  сегмент {seg['i']}: {seg['a']:.2f}..{seg['b']:.2f} "
            f"speed={seg['speed']} -> {seg['dur']:.2f} с  ({msg})")

    # --- 2. склейка concat-демуксером (без перекодирования)
    with open(os.path.join(temp, "concat.txt"), "w", encoding="utf-8") as fh:
        for f in seg_files:
            fh.write(f"file '{f}'\n")
    joined = os.path.join(temp, "joined.mov")
    r = run([FFMPEG, "-hide_banner", "-nostdin", "-y", "-f", "concat", "-safe", "0",
             "-i", "concat.txt", "-c", "copy", "joined.mov"], cwd=temp)
    if r.returncode != 0:
        raise RuntimeError("склейка не удалась:\n" + fail_tail(r))
    log(f"  склейка: {ffprobe_duration(joined):.3f} с")

    # --- 3. финальный проход
    # style:"hf" — вместо ASS-субтитров и PNG-заголовка прозрачный слой
    # HyperFrames (hf\<name>\overlay.mov); без поля — прежнее поведение
    hf_mov = None
    hf_info = None
    if edl.get("style") == "hf":
        hf_mov = run_hf_overlay(edl_path, plan)
        hf_info = hf_stats(edl_path, plan, bad, ok, hf_mov)
        ass_path, n_ev = None, 0
    else:
        ass_path, n_ev = build_ass(plan, os.path.join(temp, "subs.ass"), bad, ok)
    title_idx = None
    if plan.title and hf_mov is None:
        ti = make_title_png(plan.title, os.path.join(temp, "title.png"))
        title_idx = 1
        log(f"  заголовок: шрифт {ti['size']} px, строк {len(ti['lines'])}, "
            f"плашка {ti['pw']}x{ti['ph']} px (x {(OUT_W - ti['pw']) // 2}.."
            f"{(OUT_W - ti['pw']) // 2 + ti['pw']}, верх y={TITLE_TOP})")
    elif plan.title and hf_mov is not None:
        log(f"  заголовок: рисует слой HyperFrames ({os.path.basename(hf_mov)})")
    booms = [p["t"] for p in plan.punches if p["boom"]]
    make_boom(BOOM_WAV)
    extra_in = hf_mov if hf_mov is not None else (
        os.path.join(temp, "title.png") if title_idx is not None else None)
    boom_idx = (1 if extra_in is not None else 0) + 1
    # overlay.png (полоса + плашка) — последний вход: накладывается один раз в
    # финальном проходе, после всех зумов (punch/cam_alt) и до слоя HF/заголовка
    ovl_idx = (1 if extra_in is not None else 0) + (1 if booms else 0) + 1
    log(f"  overlay.png: {os.path.basename(ovl)} — один раз в финальном проходе "
        f"(зум панели: {'есть' if zoom_active(plan) else 'нет'}), до слоя HF/заголовка")
    # в замере входов только два (склейка + бум), поэтому индекс бума там свой
    ln = measure_loudness(plan, joined, 1 if booms else None, booms)
    log(f"  loudnorm замер: I={ln['input_i']} TP={ln['input_tp']} "
        f"LRA={ln['input_lra']} thresh={ln['input_thresh']}")
    loudnorm = (f",loudnorm=I={LOUD_I}:TP={LOUD_TP}:LRA={LOUD_LRA}"
                f":measured_I={ln['input_i']}:measured_TP={ln['input_tp']}"
                f":measured_LRA={ln['input_lra']}:measured_thresh={ln['input_thresh']}"
                f":offset={ln['target_offset']}:linear=true:print_format=summary")

    fc = build_final_filter(plan, "subs.ass", title_idx is not None, title_idx,
                            booms, boom_idx if booms else None, loudnorm,
                            hf_idx=1 if hf_mov is not None else None,
                            ovl_idx=ovl_idx)
    out_path = os.path.join(out_dir, f"{name}.mp4")
    if not os.path.isabs(out_path):
        out_path = os.path.abspath(out_path)
    check_wav = os.path.join(temp, "check_noboom.wav")
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-y", "-i", "joined.mov"]
    if extra_in is not None:
        cmd += ["-i", extra_in]
    if booms:
        cmd += ["-i", BOOM_WAV]
    if ovl_idx is not None:
        cmd += ["-i", ovl]
    cmd += ["-filter_complex", fc, "-map", "[vout]", "-map", "[aout]",
            *V_ENC, "-cq", "18", *A_ENC, "-movflags", "+faststart",
            "-t", f"{plan.total:.6f}", out_path,
            # второй выход — тот же звук после loudnorm, но без бума: по нему
            # меряются окна цензуры (бум иначе даёт ложный провал)
            "-map", "[acheck]", "-c:a", "pcm_s16le", "-ar", str(SR), "-ac", "2",
            "-t", f"{plan.total:.6f}", check_wav]
    t0 = time.time()
    if os.path.exists(out_path):
        os.remove(out_path)
    r = run(cmd, cwd=temp)
    if r.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError("финальный проход не удался:\n" + fail_tail(r, 10) +
                           f"\nфильтр: {fc}")
    log(f"  финальный проход: {time.time() - t0:.1f} с -> {out_path}")
    log(f"  звук для проверки окон (без бума): {os.path.basename(check_wav)}")

    # --- 4. проверка
    audio = audio_to_array(check_wav)
    rows = verify(plan, out_path, ass_path, n_ev, audio, hf=hf_info)
    print_table(name, rows)

    # --- 5. --sheet
    sheet_path = None
    if sheet:
        dur = ffprobe_duration(out_path)
        rows_n = max(1, math.ceil(dur / 8.0))
        sheet_path = os.path.join(out_dir, f"{name}_sheet.jpg")
        if not os.path.isabs(sheet_path):
            sheet_path = os.path.abspath(sheet_path)
        r = run([FFMPEG, "-hide_banner", "-nostdin", "-y", "-i", out_path, "-vf",
                 f"fps=1,scale=270:-1,tile=8x{rows_n}", "-frames:v", "1",
                 "-q:v", "3", sheet_path])
        if r.returncode != 0 or not os.path.exists(sheet_path):
            raise RuntimeError("не собрался sheet:\n" + fail_tail(r))
        log(f"  sheet: {sheet_path}")

    log(f"  файл: {out_path}")
    log(f"  всего на EDL: {time.time() - t_all:.1f} с")

    if not keep_temp:
        shutil.rmtree(temp, ignore_errors=True)
    else:
        log(f"  temp сохранён: {temp}")
    return all(r[3] for r in rows)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description="Динамичный монтаж вертикалок из EDL")
    ap.add_argument("edl", nargs="+", help="монтажный лист (JSON)")
    ap.add_argument("--sheet", action="store_true", help="собрать out\\<name>_sheet.jpg")
    ap.add_argument("--keep-temp", action="store_true", help="не удалять temp\\<name>")
    ap.add_argument("--out-dir", default=None,
                    help=f"куда писать <name>.mp4 (по умолчанию {OUT_DIR})")
    args = ap.parse_args()

    all_ok = True
    for edl_path in args.edl:
        try:
            ok = render_edl(edl_path, sheet=args.sheet, keep_temp=args.keep_temp,
                            out_dir=args.out_dir)
        except Exception as exc:                      # noqa: BLE001
            log(f"ОШИБКА на {edl_path}: {exc}")
            ok = False
        all_ok = all_ok and ok
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
