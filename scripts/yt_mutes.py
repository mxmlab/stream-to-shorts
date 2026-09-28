#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Окна глушения мата для YouTube-версии: Whisper по всему стриму + GigaAM.

Запуск:
    python yt_mutes.py <src> --rule letter|tt [--edl <edl.json>] [--force] [--report]

Что делает:
  1. Whisper (faster-whisper large-v3, CUDA) распознаёт `<analysis_dir>/<src>/voice.wav`
     ЦЕЛИКОМ и кладёт слова в `<src>\\whisper_words.json` (кэш; --force перезаписывает).
  2. `scripts/clips/fix_words.align` сливает слова GigaAM (`<src>\\words.json`) и Whisper ->
     `<src>\\words_merged.json`. Каждое плохое по GigaAM слово, потерянное при
     слиянии, добавляется обратно (`{w: giga, giga, start, end}`).
  3. Окна глушения — ОБЪЕДИНЕНИЕ ДВУХ ИСТОЧНИКОВ (слияние GigaAM<->Whisper
     разъезжается на повторах и галлюцинациях, по merged окна уезжают в тишину):
       источник 1 — КАЖДОЕ плохое слово GigaAM (`words.json`) по ЕГО времени;
       источник 2 — слово merged, у которого плохое только `w` (Whisper), `giga`
                    не плохое и нет плохого слова GigaAM, пересекающегося с
                    [start-0.3, end+0.3]. Если такое слово длиннее 1.2 с — окно не
                    ставится, слово идёт в отчёт как «сомнительное».
     --rule tt     — окна считает scripts\\clips\\render_tt.mute_windows_of (правило клипов);
     --rule letter — простое правило: средняя буква слова, целое слово ± padding.
  4. `<src>\\mutes_<rule>.json` — список [{"a","b","word","kind"}]; с `--edl`
     тот же список кладётся в ключ "mutes_source_time" этого EDL.

Флаги: --force  — перезаписать кэш Whisper, --report — сводка для приёмки
(источники окон, покрытие, сомнительные).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
CLIPS_DIR = os.path.join(HERE, "clips")
for _p in (REPO_ROOT, CLIPS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import render_tt                                       # noqa: E402
from config import CONFIG                              # noqa: E402

is_bad = render_tt.is_bad

_STEMS_CACHE = None


def _stems() -> tuple:
    """(bad, ok) из словарей конфига — грузятся один раз и только при первом обращении."""
    global _STEMS_CACHE
    if _STEMS_CACHE is None:
        _STEMS_CACHE = render_tt.load_stems()
    return _STEMS_CACHE


def _fix_words():
    """fix_words импортируется лениво (тянет faster-whisper/ctranslate2)."""
    if CLIPS_DIR not in sys.path:
        sys.path.insert(0, CLIPS_DIR)
    import fix_words                                   # noqa: PLC0415
    return fix_words


def _mute_windows(word: dict) -> list:
    """Окна глушения для слова — правило клипов (render_tt) на словарях конфига."""
    bad, ok = _stems()
    return render_tt.mute_windows_of(word, bad, ok)


def _parse_ranges(text: str) -> list:
    """\"6690-6705, 7083-7092\" -> [(6690.0, 6705.0), ...] (для --report)."""
    out = []
    for chunk in (text or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" not in chunk:
            raise SystemExit(f"--ranges: ожидался диапазон a-b, получено {chunk!r}")
        a, b = chunk.split("-", 1)
        out.append((float(a), float(b)))
    return out


# источник 2: допуск поиска плохого слова GigaAM рядом и порог «сомнительной» длины
PAD_NEAR = 0.3
DUBIOUS_LEN = 1.2

# Диапазоны для отчёта (--report): [(a, b), ...] в секундах исходника, заполняются
# из --ranges. Пусто — эта часть отчёта ничего не печатает.
RANGES: list = []


def log(msg: str = "") -> None:
    print(msg, flush=True)


def reconfigure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


# ================================================================ пути

def src_dir(src: str) -> str:
    """Папка исходника: как есть, если путь существует, иначе от analysis_dir."""
    if os.path.isdir(src):
        return os.path.abspath(src)
    return os.path.join(_fix_words().ANALYSIS_DIR, src)


def load_src_words(path: str) -> list:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def dump_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)


# ================================================================ 1. Whisper

def whisper_words(d: str, force: bool = False) -> tuple:
    """Слова Whisper по всему voice.wav. -> (слова, путь, секунды распознавания).

    Кэш `<src>\\whisper_words.json`; время распознавания печатается всегда
    (для кэша — 0.0, из файла время не хранится).
    """
    cache = os.path.join(d, "whisper_words.json")
    if os.path.exists(cache) and not force:
        words = load_src_words(cache)
        log(f"  Whisper: кэш {cache} — {len(words)} слов (--force чтобы распознать заново)")
        return words, cache, 0.0

    voice = os.path.join(d, "voice.wav")
    if not os.path.exists(voice):
        raise RuntimeError(f"нет {voice}")
    fw = _fix_words()
    model = fw.get_model()
    t0 = time.time()
    words = fw.transcribe_wav(model, voice)
    dt = time.time() - t0
    dump_json(cache, words)
    log(f"  Whisper: распознано {len(words)} слов за {dt:.1f} с -> {cache}")
    return words, cache, dt


# ================================================================ 2. слияние

def lost_bad_giga(giga: list, merged: list) -> list:
    """Плохие по GigaAM слова, которых нет в merged (нет слова с тем же start и giga)."""
    bad, ok = _stems()
    have = set()
    for w in merged:
        g = w.get("giga")
        if g:
            have.add((round(float(w["start"]), 3), w["w"], g))
    out = []
    for g in giga:
        if not is_bad(g["w"], bad, ok):
            continue
        key = (round(float(g["start"]), 3), g["w"], g["w"])
        if key not in have:
            out.append(g)
    return out


def fresh_start(w: dict, merged: list) -> None:
    """Сдвинуть start, пока он не свободен (совпадение ровно с start плохого слова)."""
    s = float(w["start"])
    busy = {(round(float(x["start"]), 3), x["w"], x.get("giga"))
            for x in merged if x.get("giga")}
    while (round(s, 3), w["w"], w["giga"]) in busy and s < float(w["end"]):
        s += 0.001
    w["start"] = round(s, 6)


def merge_words(d: str, giga: list, whisper: list, duration: float) -> tuple:
    """align(GigaAM, Whisper) по всему стриму. -> (merged, число добавленных)."""
    merged, _reps = _fix_words().align(giga, whisper, 0.0, duration)
    missing = lost_bad_giga(giga, merged)
    for g in missing:
        w = {"w": g["w"], "giga": g["w"],
             "start": float(g["start"]), "end": float(g["end"])}
        fresh_start(w, merged)
        merged.append(w)
    merged.sort(key=lambda x: (float(x["start"]), float(x["end"])))
    path = os.path.join(d, "words_merged.json")
    dump_json(path, merged)
    log(f"  слияние: {len(merged)} слов ({len(giga)} GigaAM + {len(whisper)} Whisper) -> {path}")
    if missing:
        log(f"  потерянных плохих слов GigaAM добавлено: {len(missing)}")
        for g in missing:
            log(f"    + {g['w']!r} {float(g['start']):.3f}..{float(g['end']):.3f}")
    else:
        log("  потерянных плохих слов GigaAM нет")
    return merged, len(missing)


# ================================================================ 3. окна

def first_bad(word: dict) -> str:
    """Первое плохое из (w, giga) — с текстом как в окне."""
    bad, ok = _stems()
    for t in (word.get("w"), word.get("giga")):
        if t and is_bad(t, bad, ok):
            return t
    return ""


def letter_windows(word: dict) -> list:
    """Простое правило: окно по средней букве; целое слово по стему — ± word_pad.

    n = len(t), m = max(1, n // 2), окно [s + d*m/n, s + d*(m+1)/n], d = end - start.
    Числа правила — из конфига, раздел `censor` (дефолты 0.03 / 0.40 / 0.15 с).
    """
    rule = CONFIG.censor()
    word_pad = float(rule["word_pad_sec"])
    window_frac = float(rule["window_frac"])
    min_window = float(rule["min_window_sec"])
    t = first_bad(word)
    if not t:
        return []
    s, e = float(word["start"]), float(word["end"])
    d = e - s
    if "пидор" in _fix_words().norm_word(t):
        return [(s - word_pad, e + word_pad, t, "word")]
    n = len(t)
    if n <= 0:
        return []
    m = max(1, n // 2)
    c = s + d * (m + 0.5) / n
    half = max(window_frac * d, min_window) / 2.0
    return [(max(s, c - half), min(e, c + half), t, "letter")]


def tt_windows(word: dict) -> list:
    """Правило клипов: окна из render_tt.mute_windows_of (те же, что в вертикалках)."""
    out = []
    for ws, we in _mute_windows(word):
        t = first_bad(word)
        if not t:
            continue
        kind = "word" if "пидор" in _fix_words().norm_word(t) else "letter"
        out.append((ws, we, t, kind))
    return out


def merge_windows(wins: list) -> list:
    """Отсортировать по (a, b) и слить пересекающиеся (тексты/виды — через запятую)."""
    wins = sorted(wins, key=lambda w: (w["a"], w["b"]))
    out = []
    for w in wins:
        if out and w["a"] <= out[-1]["b"] + 1e-9:
            last = out[-1]
            last["b"] = max(last["b"], w["b"])
            if w["word"] not in last["word"].split(","):
                last["word"] = last["word"] + "," + w["word"]
            if w["kind"] != last["kind"]:
                last["kind"] = "word" if "word" in (last["kind"], w["kind"]) else "letter"
            continue
        out.append(dict(w))
    return out


def giga_span(words: list) -> list:
    """[(start, end)] слов GigaAM — для проверки «рядом есть плохое слово»."""
    return [(float(w["start"]), float(w["end"])) for w in words]


def build_mutes(merged: list, rule: str, giga: list = None) -> tuple:
    """Окна глушения — объединение двух источников (см. докстроку модуля).

    -> (окна для mutes_<rule>.json, сведения для отчёта:
        {"wins": [{"a","b","word","kind"}],      — все окна до слияния
         "src2": [(a, b, w, giga, длина), ...],  — окна источника 2
         "doubtful": [(a, b, w, giga, длина)],   — сомнительные (> 1.2 с, без окна)
         "bad_giga": [слово GigaAM, ...]})
    """
    fn = tt_windows if rule == "tt" else letter_windows

    def word_windows(word: dict) -> list:
        out = []
        for ws, we, t, kind in fn(word):
            a, b = round(ws, 3), round(we, 3)
            if b > a:
                out.append({"a": a, "b": b, "word": t, "kind": kind})
        return out

    if giga is None:
        giga = []
    bad, ok = _stems()
    bad_giga = [g for g in giga if is_bad(g["w"], bad, ok)]

    # --- источник 1: каждое плохое слово GigaAM по ЕГО времени
    src1 = []
    for g in bad_giga:
        src1.extend(word_windows(g))

    # --- источник 2: слово merged, плохое только по `w` (Whisper)
    src2, doubtful = [], []
    for word in merged:
        if not is_bad(word.get("w", ""), bad, ok):
            continue
        if is_bad(word.get("giga", ""), bad, ok):
            continue                    # плохое слово GigaAM есть — окно даёт источник 1
        s, e = float(word["start"]), float(word["end"])
        if any(gs <= e + PAD_NEAR and ge >= s - PAD_NEAR for gs, ge in giga_span(bad_giga)):
            continue                    # рядом плохое слово GigaAM — его время вернее
        if e - s > DUBIOUS_LEN:
            doubtful.append((round(s, 3), round(e, 3), word.get("w", ""),
                             word.get("giga", ""), round(e - s, 3)))
            continue
        for w in word_windows(word):
            w["giga"] = word.get("giga", "")
            src2.append(w)
    wins = src1 + src2
    return merge_windows(wins), {"wins": wins, "src2": src2,
                                 "doubtful": doubtful, "bad_giga": bad_giga}


# ================================================================ 4. EDL

def apply_edl(path: str, mutes: list, src_name: str) -> dict:
    """Записать окна в ключ "mutes_source_time" EDL, остальные ключи не трогать."""
    if not os.path.isabs(path):
        cand = os.path.join(os.getcwd(), path)
        if not os.path.exists(cand) and not os.path.exists(path):
            cand = os.path.join(HERE, path)
        path = cand
    with open(path, encoding="utf-8") as fh:
        edl = load_src_words(path)

    dur = float(edl.get("source_duration") or 0.0)
    notes = []
    if not dur and os.path.exists(os.path.join(src_dir(src_name), "voice.wav")):
        dur = fix_words.ffprobe_duration(os.path.join(src_dir(src_name), "voice.wav"))
        notes.append("длительность взята из voice.wav")
    if dur > 0:
        n0 = len(mutes)
        mutes = [m for m in mutes if m["a"] < dur]
        for m in mutes:
            m["b"] = min(m["b"], round(dur, 3))
        mutes = merge_windows(mutes)
        if len(mutes) != n0:
            notes.append(f"окон за концом источника отброшено/слито: {n0 - len(mutes)}")

    edl["mutes_source_time"] = mutes
    dump_json(path, edl)
    log(f"  EDL: {path} — mutes_source_time: {len(mutes)} окон"
        + (f"  ({'; '.join(notes)})" if notes else ""))
    return edl


# ================================================================ разбор флагов

def resolve_edl(arg: str) -> str:
    if os.path.isabs(arg):
        return arg
    for cand in (os.path.join(os.getcwd(), arg), os.path.join(HERE, arg)):
        if os.path.exists(cand):
            return cand
    return arg


def find_edl_for(src: str) -> str:
    """EDL по имени исходника — если он есть: <src>.json, <src>\\edl.json, <база>_edit\\edl.json."""
    sd = src_dir(src)
    cands = [sd + ".json", os.path.join(sd, "edl.json")]
    base = os.path.basename(sd).rstrip("0123456789") or os.path.basename(sd)
    for name in sorted(os.listdir(sd)):
        p = os.path.join(sd, name)
        if not os.path.isdir(p):
            continue
        if not (name.endswith("_edit") or name == "tt" or name.startswith("edl")):
            continue
        for f in ("edl.json", base + "_edit.json", os.path.basename(src) + ".json"):
            q = os.path.join(p, f)
            if os.path.exists(q):
                cands.append(q)
        for f in sorted(os.listdir(p)):
            if f.endswith(".json") and base and base in f:
                cands.append(os.path.join(p, f))
    for c in cands:
        if os.path.exists(c):
            try:
                with open(c, encoding="utf-8") as fh:
                    edl = json.load(fh)
            except (OSError, ValueError):
                continue
            if edl.get("src") == src or (edl.get("source") and src in edl["source"]):
                return c
    return ""


def window_sources(mutes: list, info: dict) -> list:
    """Источник каждого окна: '1', '2' или '1+2' — по пересечению с окнами источников."""
    src1 = [w for w in info["wins"] if not w.get("giga")]
    src2 = info["src2"]
    out = []
    for m in mutes:
        one = any(w["a"] <= m["b"] + 1e-9 and w["b"] >= m["a"] - 1e-9 for w in src1)
        two = any(w["a"] <= m["b"] + 1e-9 and w["b"] >= m["a"] - 1e-9 for w in src2)
        out.append("1+2" if one and two else ("1" if one else "2"))
    return out


def check_report(mutes: list, info: dict) -> None:
    """Проверка окон: источники, диапазоны, покрытие, сомнительные."""
    sources = window_sources(mutes, info)
    n1 = sum(1 for s in sources if s == "1")
    n2 = sum(1 for s in sources if s == "2")
    n12 = sum(1 for s in sources if s == "1+2")
    lens = [(m["b"] - m["a"]) * 1000.0 for m in mutes]

    log("")
    log("-" * 78)
    log("ПРОВЕРКА ОКОН")
    if not mutes:
        log("  окон нет")
        return
    log(f"  1) окон: {len(mutes)}   источник 1: {n1}   источник 2: {n2}   "
        f"оба (1+2): {n12}")
    log(f"     наибольшая длина окна: {max(lens):.1f} мс")
    log("     5 самых длинных окон:")
    for m in sorted(mutes, key=lambda x: -(x["b"] - x["a"]))[:5]:
        log(f"       {(m['b'] - m['a']) * 1000.0:8.1f} мс  {m['a']:.3f}..{m['b']:.3f}"
            f"  {m['word']!r}  kind={m['kind']}")

    log("  2) окна по диапазонам (--ranges, по умолчанию пусто):")
    for a, b in RANGES:
        got = [(m, sources[i]) for i, m in enumerate(mutes)
               if m["a"] <= b and m["b"] >= a]
        log(f"     {a}..{b}: {len(got)}")
        for m, s in got:
            log(f"       {m['a']:.3f}..{m['b']:.3f}  {m['word']!r}  "
                f"kind={m['kind']}  источник {s}")

    uncovered = []
    for g in info["bad_giga"]:
        c = (float(g["start"]) + float(g["end"])) / 2.0
        if not any(m["a"] <= c <= m["b"] for m in mutes):
            uncovered.append((c, g["w"]))
    log(f"  3) плохих слов GigaAM: {len(info['bad_giga'])}; середина слова НЕ внутри "
        f"окна: {len(uncovered)}")
    for c, t in uncovered[:20]:
        log(f"       не покрыто: {c:.3f}  {t!r}")

    log(f"  4) сомнительные (плохое только у Whisper, слово > {DUBIOUS_LEN} с, "
        f"окно не ставилось): {len(info['doubtful'])}")
    log(f"     {'время':>17}  {'w':<14} {'giga':<14} длина_с")
    for a, b, w, g, d in info["doubtful"]:
        log(f"     {a:.3f}..{b:.3f}  {w!r:<14} {g!r:<14} {d:.3f}")


# ================================================================ 5. отчёт

def hms(t: float) -> str:
    t = max(0.0, float(t))
    h, m, s = int(t // 3600), int((t % 3600) // 60), int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


_DECODE_RE = re.compile(r"[^\w]+", re.UNICODE)


def _decode_stems() -> tuple:
    """(bad, ok) в виде, пригодном для поиска подстрокой в нормализованном тексте."""
    bad, ok = _stems()
    bad_s = [s for s in (s.strip().lower().replace("ё", "е") for s in bad) if s]
    ok_s = [s for s in (s.strip().lower().replace("ё", "е") for s in ok) if s]
    return bad_s, ok_s


def text_bad(text: str) -> bool:
    """is_bad по самим стемам (без декодирования latin-1, как в render_tt)."""
    bad_s, ok_s = _decode_stems()
    low = _DECODE_RE.sub("", (text or "").lower().replace("ё", "е"))
    if not low:
        return False
    if any(o in low for o in ok_s):
        return False
    return any(b in low for b in bad_s)


def bad_stats(giga: list, whisper: list, merged: list) -> dict:
    """Плохие слова у GigaAM / Whisper / обоих / только у одного из двух."""
    bad, ok = _stems()
    fw = _fix_words()
    sb = [w for w in giga if text_bad(w["w"])]
    wb = [w for w in whisper if text_bad(w["w"])]
    sg = {fw.norm_word(w["w"]) for w in sb}
    sw = {fw.norm_word(w["w"]) for w in wb}
    mb = [w for w in merged if is_bad(w.get("w", ""), bad, ok)
          or is_bad(w.get("giga", ""), bad, ok)]
    return {"giga": sb, "whisper": wb, "merged": mb,
            "both": sg & sw, "giga_only": sg - sw, "whisper_only": sw - sg}


def report(d: str, src: str, rule: str, mutes: list, merged: list, giga: list,
           whisper: list, added: int, whisper_dt: float, whisper_cached: bool) -> dict:
    st = bad_stats(giga, whisper, merged)
    words = [m["word"] for m in mutes]
    freq = {}
    for t in words:
        freq[t] = freq.get(t, 0) + 1
    lens = [(m["b"] - m["a"]) * 1000.0 for m in mutes]
    top = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
    n_word = sum(1 for m in mutes if m["kind"] == "word")

    log("")
    log("=" * 78)
    log(f"ОТЧЁТ: {src}  --rule {rule}   ({d})")
    log(f"  окон глушения: {len(mutes)}   из них kind=word: {n_word}")
    log((f"  средняя длина окна: {sum(lens) / len(lens):.1f} мс  "
         f"(наименьшая {min(lens):.1f}, наибольшая {max(lens):.1f})") if lens else "  окон нет")
    log("  топ-15 слов по частоте:")
    for t, n in top:
        log(f"    {n:5d}  {t!r}")
    log("  плохие слова:")
    log(f"    у GigaAM: {len(st['giga'])}")
    log(f"    у Whisper: {len(st['whisper'])}")
    log(f"    у обоих: {len(st['both'])}")
    log(f"    только у GigaAM: {len(st['giga_only'])}  "
        f"{sorted(st['giga_only'])[:24]}")
    log(f"    только у Whisper: {len(st['whisper_only'])}  "
        f"{sorted(st['whisper_only'])[:24]}")
    log(f"  слов в merged: {len(merged)}  (плохих {len(st['merged'])}, "
        f"добавлено потерянных {added})")
    log(f"  Whisper: {'кэш' if whisper_cached else f'{whisper_dt:.1f} с'}")

    rnd = random.Random(1)
    picks = rnd.sample(range(len(mutes)), min(10, len(mutes))) if mutes else []
    picks.sort()
    log("")
    log("  10 случайных окон (seed 1):")
    log(f"    {'время':>8}  {'w':<14} {'giga':<14} {'окно':<17} kind")
    for i in picks:
        m = mutes[i]
        word = next((x for x in merged
                     if x.get("w") == m["word"] or x.get("giga") == m["word"]), {})
        log(f"    {hms(m['a']):>8}  {str(word.get('w', m['word'])):<14} "
            f"{str(word.get('giga', '')):<14} "
            f"{m['a']:.3f}..{m['b']:.3f}  {m['kind']}")

    return {"rule": rule, "mutes": len(mutes), "kind_word": n_word,
            "avg_ms": (sum(lens) / len(lens)) if lens else 0.0,
            "min_ms": min(lens) if lens else 0.0,
            "max_ms": max(lens) if lens else 0.0,
            "top": top, "bad": {k: (len(v) if isinstance(v, set) else len(v))
                                for k, v in st.items()},
            "added": added, "merged": len(merged), "whisper_sec": whisper_dt}


# ================================================================ main

def main() -> int:
    reconfigure_stdout()
    ap = argparse.ArgumentParser(
        description="Whisper по всему стриму + окна глушения мата для YouTube-версии")
    ap.add_argument("src", nargs="?", default=None,
                    help="имя исходника (папка в analysis_dir) или путь к папке")
    ap.add_argument("--rule", choices=("letter", "tt"), default="letter",
                    help="letter — средняя буква слова, tt — правило вертикалок (render_tt)")
    ap.add_argument("--edl", default=None, help="EDL, куда положить mutes_source_time")
    ap.add_argument("--force", action="store_true", help="перезаписать кэш Whisper")
    ap.add_argument("--report", action="store_true", help="сводка для приёмки")
    ap.add_argument("--ranges", default="",
                    help="диапазоны для отчёта: \"a-b,c-d\" в секундах исходника")
    ap.add_argument("--edl-auto", action="store_true", help="сам найти EDL исходника")
    args = ap.parse_args()

    global RANGES
    RANGES = _parse_ranges(args.ranges)

    if not args.src:
        ap.error("укажи имя исходника (или путь к папке анализа)")

    src = args.src
    d = src_dir(src)
    if not os.path.isdir(d):
        log(f"нет папки исходника {d}")
        return 1
    t_all = time.time()

    log("=" * 78)
    log(f"ИСХОДНИК: {src}  ({d})   правило: {args.rule}")

    giga_path = os.path.join(d, "words.json")
    if not os.path.exists(giga_path):
        log(f"нет {giga_path}")
        return 1
    giga = load_src_words(giga_path)
    whisper, wpath, wdt = whisper_words(d, args.force)
    whisper_cached = wdt == 0.0

    voice = os.path.join(d, "voice.wav")
    duration = _fix_words().ffprobe_duration(voice) if os.path.exists(voice) else 0.0
    if duration <= 0:
        duration = max([float(w["end"]) for w in giga] + [float(w["end"]) for w in whisper]
                       + [0.0]) + 1.0
        log(f"  длительность voice.wav не определилась — беру {duration:.3f} с")
    else:
        log(f"  voice.wav: {duration:.3f} с")

    merged, added = merge_words(d, giga, whisper, duration)

    mutes, info = build_mutes(merged, args.rule, giga)
    out = os.path.join(d, f"mutes_{args.rule}.json")
    dump_json(out, mutes)
    log(f"  окон: {len(mutes)} -> {out}")

    edl_path = resolve_edl(args.edl) if args.edl else (
        find_edl_for(src) if args.edl_auto else "")
    if edl_path:
        apply_edl(edl_path, mutes, src)
    elif args.edl:
        log(f"  EDL не найден: {args.edl}")

    if args.report:
        report(d, src, args.rule, mutes, merged, giga, whisper, added, wdt, whisper_cached)
        check_report(mutes, info)

    log(f"  всего: {time.time() - t_all:.1f} с")
    return 0


if __name__ == "__main__":
    sys.exit(main())
