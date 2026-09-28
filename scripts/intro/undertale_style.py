# -*- coding: utf-8 -*-
"""Дисклеймер 4.5 с в стиле боевого экрана Undertale.

Кадр рисуется в 640x360 без сглаживания и увеличивается x3 (nearest) -> 1920x1080, 60 fps.
Звук синтезируется: «встреча» (мигание души), блипы печати текста, щелчки выбора кнопок.

    python undertale_style.py [--title TEXT] [--lines FILE] [--player TEXT]
                              [--out FILE] [--stills]

Заголовок, строки диалога и выходной файл задаются аргументами CLI (по умолчанию —
встроенный русский дисклеймер про ненормативную лексику). Шрифт, ffmpeg и папки —
из config.json (`font_intro`, `font_intro_body`, `ffmpeg`).
"""
import argparse
import os, subprocess, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from config import CONFIG              # noqa: E402

W, H, K = 640, 360, 3                  # низкое разрешение и множитель
FPS, DUR = 60, 4.5
NF = int(round(FPS * DUR))
SR = 48000
SEED = 7
FONT = CONFIG.font_intro
FONT_BODY = CONFIG.font_intro_body
FFMPEG = CONFIG.ffmpeg
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(CONFIG.out_dir, "intro_undertale.mp4")

WHITE, YELLOW, ORANGE, RED, BLACK = (255, 255, 255), (255, 255, 0), (255, 127, 39), (255, 0, 0), (0, 0, 0)
COLORS = {"WHITE": WHITE, "YELLOW": YELLOW, "ORANGE": ORANGE, "RED": RED, "BLACK": BLACK}

TITLE = "ДИСКЛЕЙМЕР"
# строки диалога: [(текст, цвет)], «*» только у первой, как в игре
LINES = [[("* ", WHITE), ("Данное видео", YELLOW), (" содержит", WHITE)],
         [("  ненормативную ", WHITE), ("лексику", YELLOW), (" и", WHITE)],
         [("  носит исключительно", WHITE)],
         [("  развлекательный характер.", WHITE)]]
BUTTONS = ["БИТВА", "ДЕЙСТВ", "ВЕЩИ", "ПОЩАДА"]
# надпись в HUD (имя/уровень); задаётся --player
PLAYER = "STREAMER"

# тайминг (с)
T_BLINK = (0.10, 0.55)       # мигание души
T_FLY = (0.55, 0.80)         # полёт души к кнопке
T_BOX = (0.55, 0.85)         # раскрытие рамки
T_TITLE = 0.80               # печать заголовка
TITLE_STEP = 0.06
T_TEXT = 1.20                # печать диалога
CPS = 55.0                   # символов в секунду
T_MENU = 3.45                # душа идёт по кнопкам к ПОЩАДЕ
MENU_STEP = 0.13
T_FADE = (3.95, 4.42)

HEART = [".XX...XX.",
         "XXXX.XXXX",
         "XXXXXXXXX",
         "XXXXXXXXX",
         ".XXXXXXX.",
         "..XXXXX..",
         "...XXX...",
         "....X...."]


def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def seg(t, a, b): return clamp((t - a) / (b - a))
def ease_out(x): return 1 - (1 - x) ** 3
def ease_io(x): return 0.5 - 0.5 * np.cos(np.pi * x)


F_TITLE = None
F_BODY = None
F_HUD = None
F_BTN = None


def load_fonts() -> dict:
    """Шрифты интро: заголовок, тело диалога, HUD, кнопки (пути — из конфига)."""
    return {"title": ImageFont.truetype(FONT, 50),
            "body": ImageFont.truetype(FONT_BODY, 20),
            "hud": ImageFont.truetype(FONT_BODY, 14),
            "btn": ImageFont.truetype(FONT_BODY, 16)}


def parse_lines(path: str) -> list:
    """Диалог из файла: строка = реплика; `|` делит её на куски, `@ЦВЕТ` красит кусок.

    Пример:
        * |Данное видео@YELLOW| содержит
          ненормативную |лексику@YELLOW| и
    Цвет — имя из WHITE/YELLOW/ORANGE/RED/BLACK или `@#RRGGBB`; без пометки — белый.
    """
    out = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh.read().splitlines():
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            parts = []
            for chunk in raw.split("|"):
                col = WHITE
                if "@" in chunk:
                    text, _, name = chunk.partition("@")
                    name = name.strip()
                    if name.upper() in COLORS:
                        col = COLORS[name.upper()]
                    elif name.startswith("#") and len(name) == 7:
                        col = tuple(int(name[i:i + 2], 16) for i in (1, 3, 5))
                    chunk = text
                if chunk:
                    parts.append((chunk, col))
            if parts:
                out.append(parts)
    return out

BOX = (34, 104, 606, 246)            # x0, y0, x1, y1 (внешний край)
BORDER = 3
BTN_Y = (292, 324)
BTN_X = [34 + i * 146 for i in range(4)]
BTN_W = 134
LINE_H = 30


def text_w(font, s):
    return font.getlength(s)


def draw_heart(d, cx, cy, color=RED):
    h, w = len(HEART), len(HEART[0])
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    for j, row in enumerate(HEART):
        for i, ch in enumerate(row):
            if ch == "X":
                d.point((x0 + i, y0 + j), fill=color)


def btn_heart_pos(i):
    return BTN_X[i] + 14, (BTN_Y[0] + BTN_Y[1]) / 2


def menu_index(t):
    if t < T_MENU:
        return 0
    return min(3, 1 + int((t - T_MENU) / MENU_STEP))


def typed_chars(t):
    return int(max(0.0, t - T_TEXT) * CPS)


def total_chars(lines: list | None = None) -> int:
    """Число символов диалога (по нему идёт печать и звуковые блипы)."""
    return sum(len(s) for ln in (lines if lines is not None else LINES) for s, _ in ln)


def frame(n, fonts: dict = None, title: str = None, lines: list = None,
          player: str = None):
    fonts = fonts or load_fonts()
    f_title, f_body, f_hud, f_btn = (fonts["title"], fonts["body"],
                                     fonts["hud"], fonts["btn"])
    title = TITLE if title is None else title
    lines = LINES if lines is None else lines
    player = PLAYER if player is None else player
    total = total_chars(lines)
    t = n / FPS
    rng = np.random.default_rng(SEED * 100003 + n // 3)       # дрожь меняется раз в 3 кадра
    im = Image.new("RGB", (W, H), BLACK)
    d = ImageDraw.Draw(im)
    d.fontmode = "1"

    ui = t >= T_BOX[0]
    # --- рамка диалога: раскрывается от центра по горизонтали
    if ui:
        k = ease_out(seg(t, *T_BOX))
        cx = (BOX[0] + BOX[2]) / 2
        hw = max(6, (BOX[2] - BOX[0]) / 2 * k)
        d.rectangle((int(cx - hw), BOX[1], int(cx + hw), BOX[3]), fill=WHITE)
        d.rectangle((int(cx - hw) + BORDER, BOX[1] + BORDER, int(cx + hw) - BORDER, BOX[3] - BORDER), fill=BLACK)

    # --- заголовок: печать по буквам, потом мелкая дрожь
    if t >= T_TITLE:
        nshow = min(len(title), 1 + int((t - T_TITLE) / TITLE_STEP))
        tw = text_w(f_title, title)
        x = (W - tw) / 2
        for i, ch in enumerate(title[:nshow]):
            jx, jy = (rng.integers(-1, 2), rng.integers(-1, 2)) if rng.random() < 0.35 else (0, 0)
            d.text((round(x) + jx, 22 + jy), ch, font=f_title, fill=WHITE)
            x += text_w(f_title, ch)

    # --- диалог: печать
    if t >= T_TEXT:
        left = typed_chars(t)
        y = BOX[1] + 12
        for ln in lines:
            x = BOX[0] + 18
            for s, col in ln:
                for ch in s:
                    if left <= 0:
                        break
                    jx = jy = 0
                    if col == YELLOW and t > T_TEXT + total / CPS and rng.random() < 0.3:
                        jx, jy = rng.integers(-1, 2), rng.integers(-1, 2)
                    d.text((round(x) + jx, y + jy), ch, font=f_body, fill=col)
                    x += text_w(f_body, ch)
                    left -= 1
            y += LINE_H

    # --- строка HUD и кнопки: появляются вместе с рамкой
    if ui:
        a = seg(t, T_BOX[0] + 0.1, T_BOX[1] + 0.1)
        if a > 0:
            hy = 258
            d.text((BOX[0] + 2, hy), f"{player}   LV 1", font=f_hud, fill=WHITE)
            d.text((300, hy), "HP", font=f_hud, fill=WHITE)
            d.rectangle((326, hy + 2, 326 + int(34 * a), hy + 15), fill=YELLOW)
            d.text((372, hy), "20 / 20", font=f_hud, fill=WHITE)
            sel = menu_index(t)
            for i, name in enumerate(BUTTONS):
                col = YELLOW if (i == sel and t >= T_FLY[1]) else ORANGE
                x0 = BTN_X[i]
                d.rectangle((x0, BTN_Y[0], x0 + BTN_W, BTN_Y[1]), fill=col)
                d.rectangle((x0 + 2, BTN_Y[0] + 2, x0 + BTN_W - 2, BTN_Y[1] - 2), fill=BLACK)
                tw = text_w(f_btn, name)
                d.text((x0 + 26 + (BTN_W - 26 - tw) / 2, BTN_Y[0] + 7), name, font=f_btn, fill=col)

    # --- душа
    if t < T_BLINK[0]:
        pass
    elif t < T_FLY[0]:
        if int((t - T_BLINK[0]) / 0.075) % 2 == 0:
            draw_heart(d, W / 2, H / 2)
    elif t < T_FLY[1]:
        k = ease_io(seg(t, *T_FLY))
        tx, ty = btn_heart_pos(0)
        draw_heart(d, W / 2 + (tx - W / 2) * k, H / 2 + (ty - H / 2) * k)
    else:
        draw_heart(d, *btn_heart_pos(menu_index(t)))

    arr = np.asarray(im)
    fade = 1.0 - ease_io(seg(t, *T_FADE))
    if fade < 1.0:
        arr = (arr.astype(np.float32) * fade).astype(np.uint8)
    return arr.repeat(K, 0).repeat(K, 1)


# ---------------------------------------------------------------- звук
def square(f, dur, vol, duty=0.5, decay=0.0):
    n = int(SR * dur)
    tt = np.arange(n) / SR
    w = np.where((tt * f) % 1.0 < duty, 1.0, -1.0) * vol
    env = np.ones(n)
    a = min(n, int(SR * 0.002))
    env[:a] = np.linspace(0, 1, a)
    r = min(n, int(SR * 0.006))
    env[-r:] = np.linspace(1, 0, r)
    if decay:
        env *= np.exp(-tt * decay)
    return w * env


def synth_audio(path, title: str = None, lines: list = None):
    title = TITLE if title is None else title
    total_chars_n = total_chars(lines)
    total = int(SR * DUR)
    mix = np.zeros(total)
    rng = np.random.default_rng(SEED)

    def add(sig, t0):
        i = int(t0 * SR)
        j = min(total, i + len(sig))
        if i < total:
            mix[i:j] += sig[:j - i]

    # встреча: короткий писк на каждое включение души
    k = 0
    t = T_BLINK[0]
    while t < T_FLY[0]:
        if k % 2 == 0:
            add(square(1320, 0.05, 0.22, duty=0.25), t)
        t += 0.075
        k += 1
    # шух при полёте
    add(square(660, 0.09, 0.12, duty=0.5, decay=25) + square(990, 0.09, 0.08, duty=0.5, decay=25), T_FLY[0])
    # печать заголовка: низкий блип на букву
    for i in range(len(title)):
        add(square(330, 0.045, 0.2, duty=0.5, decay=20), T_TITLE + i * TITLE_STEP)
    # печать диалога: блип на каждую вторую букву, чуть плавает высота
    for i in range(0, total_chars_n, 2):
        add(square(560 + rng.integers(-15, 16), 0.035, 0.16, duty=0.5, decay=30), T_TEXT + i / CPS)
    # переходы по кнопкам
    for i in range(3):
        add(square(880, 0.04, 0.2, duty=0.25, decay=40), T_MENU + i * MENU_STEP)
    # выбор ПОЩАДЫ: двойной тон
    add(square(1175, 0.06, 0.2, duty=0.5, decay=15), T_MENU + 3 * MENU_STEP)
    add(square(1568, 0.09, 0.18, duty=0.5, decay=12), T_MENU + 3 * MENU_STEP + 0.06)

    # общий уход вместе с картинкой
    tt = np.arange(total) / SR
    mix *= np.array([1.0 - ease_io(seg(x, *T_FADE)) for x in tt[::480]]).repeat(480)[:total]
    mix = np.clip(mix * 0.8, -0.95, 0.95)
    st = np.stack([mix, mix], 1)
    pcm = (st * 32767).astype("<i2")
    import wave
    with wave.open(path, "wb") as wv:
        wv.setnchannels(2); wv.setsampwidth(2); wv.setframerate(SR)
        wv.writeframes(pcm.tobytes())


def stills(fonts: dict, title: str, lines: list, player: str, out_dir: str = None):
    out_dir = out_dir or HERE
    ts = [0.2, 0.7, 1.0, 1.5, 2.2, 3.0, 3.5, 3.9, 4.3]
    ims = [Image.fromarray(frame(int(x * FPS), fonts, title, lines, player)).resize((640, 360), Image.NEAREST)
           for x in ts]
    sheet = Image.new("RGB", (640 * 3 + 20, 360 * 3 + 20), (40, 40, 40))
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % 3) * 650, (i // 3) * 370))
    sheet.save(os.path.join(out_dir, "intro_preview.jpg"), quality=90)
    Image.fromarray(frame(int(3.8 * FPS), fonts, title, lines, player)).save(
        os.path.join(out_dir, "intro_poster.png"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="undertale_style.py",
        description="Дисклеймер 4.5 с в стиле боевого экрана Undertale")
    ap.add_argument("--title", default=TITLE, help=f"заголовок (по умолчанию {TITLE!r})")
    ap.add_argument("--lines", default=None, metavar="FILE",
                    help="файл строк диалога (строка = реплика, `|` делит, `@ЦВЕТ` красит)")
    ap.add_argument("--player", default=PLAYER, help="надпись в HUD (имя/уровень)")
    ap.add_argument("--out", default=OUT, help=f"выходной mp4 (по умолчанию {OUT})")
    ap.add_argument("--stills", action="store_true", help="только превью-сетка и постер, без видео")
    ap.add_argument("--preview-dir", default=None, help="куда класть превью (по умолчанию рядом со скриптом)")
    args = ap.parse_args(argv)

    t0 = time.time()
    fonts = load_fonts()
    lines = parse_lines(args.lines) if args.lines else LINES
    if not lines:
        raise SystemExit("--lines: в файле нет ни одной строки диалога")
    out = os.path.normpath(args.out)
    stills(fonts, args.title, lines, args.player, args.preview_dir)
    if args.stills:
        print(f"превью готово за {time.time() - t0:.1f} с")
        return 0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    wav = os.path.join(CONFIG.temp_dir, "intro_sfx.wav")
    os.makedirs(CONFIG.temp_dir, exist_ok=True)
    synth_audio(wav, args.title, lines)
    cmd = [FFMPEG, "-y", "-nostdin", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W*K}x{H*K}", "-r", str(FPS), "-i", "-",
           "-i", wav,
           "-c:v", "libx264", "-profile:v", "high", "-crf", "14", "-preset", "slow", "-pix_fmt", "yuv420p",
           "-tune", "animation", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
           "-frames:v", str(NF), "-shortest", "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for n in range(NF):
        p.stdin.write(frame(n, fonts, args.title, lines, args.player).tobytes())
    p.stdin.close()
    p.wait()
    print(f"{out}  за {time.time() - t0:.1f} с, код {p.returncode}")
    return 0 if p.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
