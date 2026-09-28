#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tighten.py — динамичный перемонтаж роликов: паузы -> склейки, крючок в начале.

Вход:   <in>/*.json + <in>/<прежний name>.words.json,
        <features_dir>/<src>/features.csv,       (features_dir — из конфига,
        <out>/hooks.json, <out>/visual.json.      по умолчанию analysis_dir)
Выход:  <out>/<то же имя>.json            (name = "tt_" + прежний без "exp_",
                                           если прежний начинался с "exp_"; иначе без правки),
        <out>/<новый name>.words.json       (побайтовая копия файла слов).

Папки <in>/<out> задаются аргументами --in / --out (относительные — от папки
скрипта), по умолчанию edl_exp / edl_exp2. hooks.json и visual.json читаются из
<out>; если файла нет — он считается пустым ({} / []).

Правила:
  1. речевой остров = слово ± PAD; острова с зазором ≤ MERGE_GAP сливаются;
  2. громкая секунда [t, t+1) (voice_db > VOICE_DB или scream > SCREAM или
     laugh > LAUGH) — тоже остров (крик/смех без слов не режем);
  3. зазор между островами > CUT_GAP: середина вырезается, но между островами
     остаётся KEEP (по KEEP/2 с каждой стороны). Мягкий режим (visual.json):
     зазор сокращается до SOFT_KEEP (по SOFT_KEEP/2 с каждой стороны), а зазоры
     ≤ SOFT_CUT_GAP не трогаются вовсе (остаются в ролике целиком);
  4. начало/конец сегмента подрезаются по первому/последнему острову,
     но не дальше границ сегмента;
  5. куски короче MIN_PIECE слить с соседним (вместе с зазором между ними);
  6. крючок из hooks.json (если не null) — первым сегментом, до основного
     монтажа; повторение этого места в основном монтаже остаётся.

Запуск:  python tighten.py [--in <папка>] [--out <папка>]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent                 # корень репозитория
sys.path.insert(0, str(REPO))
from config import CONFIG                 # noqa: E402

ANALYSIS = Path(CONFIG.features_dir)      # <features_dir>/<src>/features.csv
DIR_IN = HERE / "edl_exp"         # переопределяется --in
DIR_OUT = HERE / "edl_exp2"       # переопределяется --out

# --- параметры правил ------------------------------------------------------
PAD = 0.12            # п.1: слово ± 0.12 с
MERGE_GAP = 0.45      # п.1: острова с зазором ≤ 0.45 с сливаются
CUT_GAP = 0.45        # п.3: обычный режим — зазор > 0.45 с режется
KEEP = 0.25           # п.3: ... но 0.25 с между островами остаётся (0.125 + 0.125)
SOFT_CUT_GAP = 0.60   # п.3: мягкий режим — зазоры ≤ 0.6 с не трогать
SOFT_KEEP = 0.60      # п.3: ... а > 0.6 с укорачивать до 0.6 (0.3 + 0.3)
MIN_PIECE = 0.25      # п.5: кусок короче 0.25 с — к соседу
VOICE_DB = -32.0      # п.2: громкая секунда
SCREAM = 0.35
LAUGH = 0.20
PUNCH_EPS = 1e-3      # сдвиг punch.t внутрь сегмента
EPS = 1e-9
ND = 3                # знаков после запятой в EDL


def rnd(x: float) -> float:
    return round(float(x) + 0.0, ND)


# ============================================================ вход


def load_hooks() -> dict:
    p = DIR_OUT / "hooks.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return data or {}


def load_visual() -> set:
    p = DIR_OUT / "visual.json"
    if not p.exists():
        return set()
    data = json.loads(p.read_text(encoding="utf-8"))
    return set(data or [])


_loud_cache: dict = {}


def loud_seconds(src: str) -> list:
    """Секунды t (время исходника), где voice_db > -32 или scream > 0.35 или laugh > 0.2."""
    if src in _loud_cache:
        return _loud_cache[src]
    path = ANALYSIS / src / "features.csv"
    if not path.exists():
        raise SystemExit(f"нет файла признаков: {path}")
    out = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                t = int(float(row["t"]))
                v = float(row["voice_db"])
                sc = float(row["scream"])
                la = float(row["laugh"])
            except (TypeError, ValueError, KeyError):
                continue
            if v > VOICE_DB or sc > SCREAM or la > LAUGH:
                out.append(t)
    out.sort()
    _loud_cache[src] = out
    return out


def load_words(path: Path) -> list:
    words = json.loads(path.read_text(encoding="utf-8"))
    return [(float(w["start"]), float(w["end"])) for w in words]


def new_name(name: str) -> str:
    """exp_* -> tt_*; прочие имена не меняются."""
    return "tt_" + name[4:] if name.startswith("exp_") else name


def resolve_dir(p: str) -> Path:
    """Относительный путь — от папки скрипта, абсолютный — как есть."""
    q = Path(p)
    return q if q.is_absolute() else (HERE / q)


# ============================================================ правила


def build_islands(a: float, b: float, words: list, loud: list) -> list:
    """Острова внутри сегмента [a, b]: слово ± PAD и громкие секунды, слитые по зазору ≤ 0.45."""
    iv = []
    for ws, we in words:
        lo, hi = max(a, ws - PAD), min(b, we + PAD)
        if hi > lo:
            iv.append([lo, hi])
    # секунда [t, t+1) пересекается с [a, b)  <=>  t > a-1 и t < b
    for t in loud[bisect_right(loud, a - 1.0):bisect_left(loud, b)]:
        lo, hi = max(a, float(t)), min(b, float(t) + 1.0)
        if hi > lo:
            iv.append([lo, hi])
    if not iv:
        return []
    iv.sort()
    merged = [iv[0]]
    for lo, hi in iv[1:]:
        if lo - merged[-1][1] <= MERGE_GAP + EPS:      # п.1: зазор ≤ 0.45 — сливаем
            if hi > merged[-1][1]:
                merged[-1][1] = hi
        else:
            merged.append([lo, hi])
    return [(round(x, 6), round(y, 6)) for x, y in merged]


def keep_pieces(islands: list, soft: bool) -> list:
    """п.3+п.4: куски, которые остаются в ролике (в границах сегмента [a, b])."""
    if not islands:
        return []
    cut_gap = SOFT_CUT_GAP if soft else CUT_GAP
    half = (SOFT_KEEP if soft else KEEP) / 2.0
    out = [[islands[0][0], islands[0][1]]]
    for i in range(1, len(islands)):
        s, e = islands[i]
        gap = s - islands[i - 1][1]
        if gap <= cut_gap + EPS:
            # зазор не режем — материал остаётся, кусок тянется дальше
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            # режем: с каждой стороны остаётся по half
            out[-1][1] = islands[i - 1][1] + half
            out.append([s - half, e])
    return out


def merge_short(pieces: list) -> list:
    """п.5: куски короче 0.25 с — к соседу вместе с зазором между ними."""
    out = [list(p) for p in pieces]
    changed = True
    while changed and len(out) > 1:
        changed = False
        for i, p in enumerate(out):
            if p[1] - p[0] < MIN_PIECE - EPS:
                if i > 0:
                    out[i - 1][1] = max(out[i - 1][1], p[1])
                    del out[i]
                else:
                    out[i + 1][0] = min(out[i + 1][0], p[0])
                    del out[i]
                changed = True
                break
    return out


def punches_inside(segments: list, t: float) -> bool:
    return any(s["a"] - EPS <= t < s["b"] for s in segments)


def _nearest(segments: list, t: float) -> dict:
    def dist(s):
        if t < s["a"]:
            return s["a"] - t
        if t >= s["b"]:
            return t - s["b"] + PUNCH_EPS
        return 0.0
    return min(segments, key=dist)


def _move_inside(seg: dict, t: float) -> float:
    """Ближайшая к t точка внутри [a, b) с округлением до 3 знаков."""
    if t < seg["a"]:
        return rnd(seg["a"])
    if t >= seg["b"]:
        nt = rnd(seg["b"] - PUNCH_EPS)
        if nt >= seg["b"]:
            nt = rnd(seg["b"] - 0.01)
        return nt
    return rnd(t)


# ============================================================ обработка ролика


def tighten(edl: dict, words: list, loud: list, soft: bool, hook) -> tuple:
    """Возвращает (новый EDL, [(i, было, стало)] сдвинутых punch, [пустые сегменты])."""
    main = []
    empty = []
    for si, seg in enumerate(edl.get("segments") or []):
        a, b = float(seg["a"]), float(seg["b"])
        seg_words = [(ws, we) for ws, we in words
                     if ws >= a - EPS and we <= b + EPS]        # слова целиком внутри
        pieces = merge_short(keep_pieces(build_islands(a, b, seg_words, loud), soft))
        if not pieces:
            empty.append(si)
        for lo, hi in pieces:
            lo, hi = max(lo, a), min(hi, b)
            lo, hi = rnd(lo), rnd(hi)
            if hi - lo > EPS:
                main.append({"a": lo, "b": hi})

    segments = []
    if hook:
        segments.append({"a": rnd(hook[0]), "b": rnd(hook[1])})
    segments.extend(main)

    out = dict(edl)
    out["name"] = new_name(edl["name"])
    out["segments"] = segments

    shifted = []
    if edl.get("punch"):
        punches = []
        for i, p in enumerate(edl["punch"]):
            q = dict(p)
            t = float(q["t"])
            if segments and not punches_inside(segments, t):
                nt = _move_inside(_nearest(segments, t), t)
                shifted.append((i, t, nt))
                q["t"] = nt
            punches.append(q)
        out["punch"] = punches
    return out, shifted, empty


# ============================================================ проверка и таблица


def verify(src_edl: dict, out_edl: dict, words: list) -> dict:
    """Проверка по записанному EDL: punch внутри сегментов, потерянных слов нет."""
    segs = [(float(s["a"]), float(s["b"])) for s in out_edl["segments"]]

    bad_punch = [float(p["t"]) for p in (out_edl.get("punch") or [])
                 if not punches_inside(out_edl["segments"], float(p["t"]))]

    # слова исходных сегментов (целиком внутри сегмента) — все должны остаться
    src_words = [(ws, we) for ws, we in words
                 if any(ws >= a - EPS and we <= b + EPS for a, b in
                        [(float(s["a"]), float(s["b"])) for s in src_edl["segments"]])]
    lost = []
    for ws, we in src_words:
        if not any(a - EPS <= ws and we <= b + EPS for a, b in segs):
            lost.append((ws, we))

    # слова, которые не принадлежат ни одному исходному сегменту целиком
    straddle = len(words) - len(src_words)

    return {"bad_punch": bad_punch, "words": len(src_words), "lost": lost,
            "straddle": straddle, "segments": len(segs)}


def fmt(x: float, n: int = 2) -> str:
    return f"{x:.{n}f}"


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="tighten.py",
        description="Динамичный перемонтаж роликов: паузы -> склейки, крючок в начале.")
    ap.add_argument("--in", dest="dir_in", default="edl_exp", metavar="<папка>",
                    help="папка с входными EDL и файлами слов (по умолчанию edl_exp)")
    ap.add_argument("--out", dest="dir_out", default="edl_exp2", metavar="<папка>",
                    help="папка вывода и чтения hooks.json/visual.json (по умолчанию edl_exp2)")
    return ap.parse_args(argv)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    global DIR_IN, DIR_OUT
    args = parse_args()
    DIR_IN = resolve_dir(args.dir_in)
    DIR_OUT = resolve_dir(args.dir_out)
    DIR_OUT.mkdir(parents=True, exist_ok=True)

    hooks = load_hooks()
    visual = load_visual()
    files = sorted(p for p in DIR_IN.glob("*.json") if not p.name.endswith(".words.json"))
    if not files:
        raise SystemExit(f"нет EDL в {DIR_IN}")

    rows, fails, notes = [], [], []
    tot_before = tot_after = tot_hook = 0.0
    tot_pieces = tot_lost = tot_words = 0

    for path in files:
        edl = json.loads(path.read_text(encoding="utf-8"))
        name = edl["name"]
        wpath = DIR_IN / (name + ".words.json")
        if not wpath.exists():
            raise SystemExit(f"нет файла слов: {wpath}")
        words = load_words(wpath)
        soft = path.stem in visual
        hook = hooks.get(path.stem)
        if hook is not None:
            hook = (float(hook[0]), float(hook[1]))

        out, shifted, empty = tighten(edl, words, loud_seconds(edl["src"]), soft, hook)
        if not out["segments"]:
            raise SystemExit(f"{path.name}: после уплотнения не осталось сегментов")
        if empty:
            notes.append(f"{path.name}: исходные сегменты без островов (выпали): {empty}")

        dst = DIR_OUT / path.name
        with dst.open("w", encoding="utf-8") as f:            # как в edl_exp: indent=1, CRLF
            json.dump(out, f, ensure_ascii=False, indent=1)
        shutil.copyfile(wpath, DIR_OUT / (out["name"] + ".words.json"))

        # --- проверяем уже записанный файл
        written = json.loads(dst.read_text(encoding="utf-8"))
        chk = verify(edl, written, words)

        before = sum(float(s["b"]) - float(s["a"]) for s in edl["segments"])
        after = sum(float(s["b"]) - float(s["a"]) for s in written["segments"])
        hook_len = (float(written["segments"][0]["b"]) - float(written["segments"][0]["a"])
                    if hook else 0.0)
        pieces = len(written["segments"]) - (1 if hook else 0)
        cut = 100.0 * (1.0 - (after - hook_len) / before) if before else 0.0

        rows.append({
            "file": path.name, "name": out["name"], "mode": "мягкий" if soft else "обычный",
            "before": before, "after": after, "hook": hook_len, "pieces": pieces,
            "cut": cut, "shifted": shifted, "lost": len(chk["lost"]),
            "words": chk["words"], "straddle": chk["straddle"],
            "bad_punch": chk["bad_punch"],
        })
        tot_before += before
        tot_after += after
        tot_hook += hook_len
        tot_pieces += pieces
        tot_words += chk["words"]
        tot_lost += len(chk["lost"])
        if chk["bad_punch"]:
            fails.append(f"{path.name}: punch вне сегментов {chk['bad_punch']}")
        if chk["lost"]:
            fails.append(f"{path.name}: потеряно слов {len(chk['lost'])}: {chk['lost'][:5]}")
        if shifted:
            notes.append(f"{path.name}: сдвинут punch {shifted}")

    tot_cut = 100.0 * (1.0 - (tot_after - tot_hook) / tot_before) if tot_before else 0.0

    print(f"# tighten.py — {len(rows)} роликов -> {DIR_OUT}")
    print()
    print("| # | файл | режим | было, с | стало, с | крючок, с | кусков | вырезано, % | punch | потеряно слов |")
    print("|---|------|-------|--------:|---------:|----------:|-------:|------------:|-------|--------------:|")
    for i, r in enumerate(rows, 1):
        punch = "ok" if not r["shifted"] else "сдвинут " + ", ".join(
            f"{o:.2f}->{n:.2f}" for _, o, n in r["shifted"])
        print(f"| {i} | `{r['file']}` → `{r['name']}` | {r['mode']} | {fmt(r['before'])} | "
              f"{fmt(r['after'])} | {fmt(r['hook'])} | {r['pieces']} | {fmt(r['cut'], 1)} | "
              f"{punch} | {r['lost']} |")
    print(f"| — | **итого {len(rows)}** | | **{fmt(tot_before)}** | **{fmt(tot_after)}** | "
          f"**{fmt(tot_hook)}** | **{tot_pieces}** | **{fmt(tot_cut, 1)}** | | **{tot_lost}** |")
    print()
    print(f"проверено слов (целиком внутри исходных сегментов): {tot_words}; "
          f"потеряно: {tot_lost}")
    print(f"слова, не попадающие в исходные сегменты целиком (вне проверки): "
          f"{sum(r['straddle'] for r in rows)}")
    print(f"punch вне сегментов после записи: {sum(len(r['bad_punch']) for r in rows)}; "
          f"сдвинуто при обработке: {sum(len(r['shifted']) for r in rows)}")
    if fails or notes:
        print()
        print("!! замечания:")
        for m in fails + notes:
            print("  -", m)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
