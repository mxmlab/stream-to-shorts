#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Текст субтитров — из Whisper, время — из GigaAM.

Запуск:
    python fix_words.py <edl.json> [<edl2.json> ...] [--force]

Для каждого сегмента EDL:
  1. из `<analysis_dir>\\<src>\\voice.wav` (голос без игры, 16 кГц моно) вырезается
     кусок [a-0.3, b+0.3] во временный wav;
  2. кусок распознаёт faster-whisper large-v3 ровно как в заимствованном коде
     (см. строки атрибуции в шапке файла): language="ru",
     word_timestamps=True, vad_filter=True, condition_on_previous_text=False,
     no_speech_threshold=0.6; времена Whisper переводятся во время исходника
     (+ начало куска);
  3. слова Whisper выравниваются со словами GigaAM (start ∈ [a, b))
     через difflib.SequenceMatcher по нормализованным словам (нижний регистр,
     ё→е, только буквы/цифры);
  4. результат — `<папка EDL>\\<name>.words.json` ([{"w","start","end","src"}], src =
     eq|rep|ins|del|giga) и `<папка EDL>\\<name>.words.txt` (по сегментам: строка GigaAM
     и строка итоговая). В блоке delete остаётся короткое междометие (<= 4 букв)
     и любое плохое по is_bad слово: Whisper теряет мат, и без этого слово
     выпадает и из субтитров, и из цензуры.

Кэш: если .words.json новее EDL — сегмент пропускается (перебить: --force).
Ручная правка .words.json поэтому не затирается.

# Adapted from Reelsi (https://github.com/mxmlab/reelsi), AGPL-3.0, reelsi/core/transcribe.py
# Adapted from Reelsi (https://github.com/mxmlab/reelsi), AGPL-3.0, reelsi/core/cuda_env.py
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import site
import subprocess
import sys
import time

TT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(TT_DIR))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
from config import CONFIG                            # noqa: E402

STREAM_ROOT = CONFIG.stream_root
ANALYSIS_DIR = CONFIG.analysis_dir
TEMP_ROOT = os.path.join(CONFIG.temp_dir, "fix_words")

if TT_DIR not in sys.path:
    sys.path.insert(0, TT_DIR)
import render_tt                                    # noqa: E402  (правила цензуры)

FFMPEG = CONFIG.ffmpeg
FFPROBE = CONFIG.ffprobe

PAD = 0.3                        # с, запас куска вокруг сегмента
MODEL_SIZE = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
LANG = "ru"


def log(msg: str = "") -> None:
    print(msg, flush=True)


def run(cmd) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace")


def fail_tail(r: subprocess.CompletedProcess, n: int = 6) -> str:
    tail = "\n".join((r.stderr or "").strip().splitlines()[-n:])
    return f"ffmpeg rc={r.returncode}\n{tail}"


# ================================================================ CUDA DLL
# Копия setup() из заимствованного модуля (см. строки атрибуции в шапке файла).
# Вызывается ДО импорта faster_whisper, иначе ctranslate2 не найдёт cublas/cudnn.


def setup_cuda_dlls() -> list:
    bases = set(site.getsitepackages() + [site.getusersitepackages()])
    dirs = []
    for base in bases:
        for sub in ("cublas", "cudnn", "cuda_runtime"):
            d = os.path.join(base, "nvidia", sub, "bin")
            if os.path.isdir(d):
                dirs.append(d)
    if dirs:
        os.environ["PATH"] = os.pathsep.join(dirs) + os.pathsep + os.environ.get("PATH", "")
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if add_dll_directory is not None:
            for d in dirs:
                try:
                    add_dll_directory(d)
                except OSError:
                    pass  # каталог не принят как DLL-путь — PATH уже дополнен выше
    return dirs


CUDA_DLL_DIRS = setup_cuda_dlls()

# ================================================================ Whisper как в заимствованном коде
# Фильтры галлюцинаций — копия transcribe.py (строки 95-142),
# сам вызов transcribe() — копия строк 145-162 (пути — в строках атрибуции шапки).

CREDITS = [
    "субтитры делал", "субтитры сделал", "субтитры создавал", "субтитры подготовил",
]
CREDITS_NAMED = re.compile(
    r"(?i:редактор\s+субтитров|корректор)\s+[A-ZА-ЯЁ]\.\s*[A-ZА-ЯЁ]\w*"
)
CREDITS_BARE = [
    "редактор субтитров", "корректор",
]
CTA_BOILER = [
    "продолжение следует", "спасибо за просмотр", "спасибо за внимание",
    "подписывайтесь", "ставьте лайки", "до новых встреч", "продолжение в следующей",
    "subscribe", "thanks for watching",
]
HALLUCINATIONS = CREDITS + CREDITS_BARE + CTA_BOILER


def _drop_segment(s) -> bool:
    raw = s.text or ""
    t = raw.lower()
    nsp = getattr(s, "no_speech_prob", 0.0)
    alp = getattr(s, "avg_logprob", 0.0)
    if any(h in t for h in CREDITS) or bool(CREDITS_NAMED.search(raw)):
        return True
    if nsp > 0.7 and alp < -0.6:                       # детектор галлюцинаций на не-речи
        return True
    low_conf = (nsp > 0.5 or alp < -0.7)
    if any(h in t for h in CTA_BOILER) and low_conf:
        return True
    if any(h in t for h in CREDITS_BARE) and low_conf:
        return True
    return False


_MODEL = None
MODEL_LOAD_TIME = 0.0


def get_model():
    """Модель грузится ОДИН раз на весь прогон (кэш в модуле)."""
    global _MODEL, MODEL_LOAD_TIME
    if _MODEL is None:
        from faster_whisper import WhisperModel      # импорт здесь: ctranslate2 не найдёт cublas/cudnn,
        t0 = time.time()                             # если сделать его раньше, чем setup_cuda_dlls()
        _MODEL = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        MODEL_LOAD_TIME = time.time() - t0
        log(f"  Whisper {MODEL_SIZE} ({DEVICE}/{COMPUTE_TYPE}): модель загружена "
            f"за {MODEL_LOAD_TIME:.1f} с")
    return _MODEL


def transcribe_wav(model, wav_path: str) -> list:
    """Ровно как transcribe.py заимствованного модуля, строки 145-162."""
    segments, info = model.transcribe(
        wav_path, language=LANG, word_timestamps=True,
        vad_filter=True,                    # skip non-speech -> kills most hallucinations
        condition_on_previous_text=False,   # stop runaway repetition loops
        no_speech_threshold=0.6)
    words = []
    for s in segments:
        if _drop_segment(s):
            continue
        for w in (s.words or []):
            t = w.word.strip()
            if t:
                words.append({"w": t, "start": float(w.start), "end": float(w.end)})
    return words


# ================================================================ нормализация / выравнивание

_norm_re = re.compile(r"[^0-9a-zа-я]")     # после lower() и ё→е
_clean_re = re.compile(r"[^0-9a-zа-яё]")


def norm_word(s: str) -> str:
    """Нормализация ДЛЯ СРАВНЕНИЯ: нижний регистр, ё→е, только буквы/цифры."""
    return _norm_re.sub("", (s or "").lower().replace("ё", "е"))


def clean_word(s: str) -> str:
    """Текст слова в результат: без пунктуации, нижний регистр."""
    return _clean_re.sub("", (s or "").lower())


SWAPS = []          # [(слово Whisper, слово GigaAM)] — где Whisper «почистил» мат


def _w(text: str, start, end, src: str, giga: str = None) -> dict:
    d = {"w": text, "start": round(float(start), 6), "end": round(float(end), 6),
         "src": src}
    if giga is not None:
        d["giga"] = giga
    return d


_BAD_OK: tuple | None = None


def _bad_ok() -> tuple:
    """(bad, ok) из словарей конфига; грузятся один раз и только при первом обращении."""
    global _BAD_OK
    if _BAD_OK is None:
        _BAD_OK = render_tt.load_stems()
    return _BAD_OK


def _pair(whisper_text: str, giga_word: str, start, end, src: str) -> dict:
    """Слово результата: текст Whisper, исходное слово GigaAM — в поле `giga`.

    Если слово GigaAM плохое (is_bad по правилам render_tt), а слово Whisper —
    нет, в `w` остаётся слово GigaAM: Whisper «чистит» мат («бля» -> «для»), и
    по одному `w` цензура его не находит. В субтитрах такое слово идёт со
    звёздочкой, а звук глушится (render_tt смотрит и `w`, и `giga`).
    """
    bad, ok = _bad_ok()
    w = clean_word(whisper_text)
    g = clean_word(giga_word) if giga_word else ""
    if (g and render_tt.is_bad(g, bad, ok)
            and not render_tt.is_bad(w, bad, ok)):
        SWAPS.append((w, g))
        w = g
    return _w(w, start, end, src, giga=g)


def align(g_words: list, w_words: list, a: float, b: float) -> tuple:
    """Выравнивание слов GigaAM (start ∈ [a,b)) со словами Whisper.

    -> (слова результата, примеры replace [(было, стало), ...]).
    """
    gn = [norm_word(g["w"]) for g in g_words]
    wn = [norm_word(w["w"]) for w in w_words]
    sm = difflib.SequenceMatcher(None, gn, wn, autojunk=False)
    out, reps = [], []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            # текст Whisper, время GigaAM 1:1; giga — то же слово
            for k in range(i2 - i1):
                g = g_words[i1 + k]
                out.append(_pair(w_words[j1 + k]["w"], w_words[j1 + k]["w"],
                                 g["start"], g["end"], "eq"))
        elif tag == "replace":
            # текст Whisper; время — интервал GigaAM-блока, поделённый по числу букв
            blk = w_words[j1:j2]
            gs = float(g_words[i1]["start"])
            ge = float(g_words[i2 - 1]["end"])
            if len(blk) == (i2 - i1):
                for k, wp in enumerate(blk):        # слов поровну — 1:1 по времени GigaAM
                    g = g_words[i1 + k]
                    out.append(_pair(wp["w"], g["w"], g["start"], g["end"], "rep"))
            else:
                weights = [len(norm_word(wp["w"])) for wp in blk]
                if sum(weights) <= 0:
                    weights = [1] * len(blk)
                total = float(sum(weights))
                acc = 0.0
                for k, wp in enumerate(blk):
                    frac = acc / total
                    s = gs + (ge - gs) * frac
                    acc += weights[k]
                    e = gs + (ge - gs) * acc / total
                    # GigaAM-блок без 1:1 — исходное слово берём по доле времени
                    gi = i1 + min(int(frac * (i2 - i1)), i2 - i1 - 1)
                    out.append(_pair(wp["w"], g_words[gi]["w"], s, e, "rep"))
            reps.append((" ".join(g["w"] for g in g_words[i1:i2]),
                         " ".join(clean_word(wp["w"]) for wp in blk)))
        elif tag == "insert":
            # слово Whisper со своим временем, обрезанным в [a, b)
            for wp in w_words[j1:j2]:
                s = min(max(float(wp["start"]), a), b)
                e = min(max(float(wp["end"]), a), b)
                out.append(_w(clean_word(wp["w"]), s, max(e, s), "ins"))
        else:                                       # delete
            # ОСТАЁТСЯ короткое междометие (<= 4 букв) и ЛЮБОЕ плохое слово:
            # Whisper теряет мат («пиздец»), и без этого слово выпадает сразу
            # и из субтитров, и из цензуры. Пометка src='del'.
            for g in g_words[i1:i2]:
                bad, ok = _bad_ok()
                if (1 <= len(norm_word(g["w"])) <= 4
                        or render_tt.is_bad(g["w"], bad, ok)):
                    out.append(_pair(g["w"], g["w"], g["start"], g["end"], "del"))
    return out, reps


# ================================================================ файлы

def source_words_path(src: str) -> str:
    return os.path.join(ANALYSIS_DIR, src, "words.json")


def voice_path(src: str) -> str:
    return os.path.join(ANALYSIS_DIR, src, "voice.wav")


def ffprobe_duration(path: str) -> float:
    r = run([FFPROBE, "-hide_banner", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path])
    try:
        return float(r.stdout.strip())
    except ValueError:
        return -1.0


def cut_chunk(voice: str, a: float, b: float, dst: str, voice_dur: float) -> float:
    """Вырезать [a-0.3, b+0.3] в 16 кГц моно wav. -> время начала куска в исходнике."""
    s = max(0.0, a - PAD)
    e = b + PAD
    if voice_dur > 0:
        e = min(e, voice_dur)
    d = max(e - s, 0.01)
    cmd = [FFMPEG, "-hide_banner", "-nostdin", "-y",
           "-ss", f"{s:.6f}", "-t", f"{d:.6f}", "-i", voice,
           "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", dst]
    r = run(cmd)
    if r.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError(f"не вырезался кусок {a}..{b}:\n{fail_tail(r)}")
    return s


def process_edl(edl_path: str, force: bool) -> dict:
    """Обработать один EDL. -> статистика (или {'skipped': True})."""
    with open(edl_path, encoding="utf-8") as fh:
        edl = json.load(fh)
    name = edl["name"]
    src = edl["src"]
    edl_dir = os.path.dirname(os.path.abspath(edl_path))
    out_json = os.path.join(edl_dir, f"{name}.words.json")
    out_txt = os.path.join(edl_dir, f"{name}.words.txt")

    log("=" * 78)
    log(f"EDL: {edl_path}")
    if (not force and os.path.exists(out_json)
            and os.path.getmtime(out_json) > os.path.getmtime(edl_path)):
        log(f"  {os.path.basename(out_json)} новее EDL — пропуск (--force чтобы переписать)")
        return {"name": name, "skipped": True}

    voice = voice_path(src)
    if not os.path.exists(voice):
        raise RuntimeError(f"нет {voice}")
    with open(source_words_path(src), encoding="utf-8") as fh:
        giga = json.load(fh)
    voice_dur = ffprobe_duration(voice)

    segments = edl.get("segments") or []
    temp = os.path.join(TEMP_ROOT, "fix_words", name)
    os.makedirs(temp, exist_ok=True)
    model = get_model()

    t0 = time.time()
    all_words, per_seg = [], []
    stats = {"eq": 0, "rep": 0, "ins": 0, "del": 0, "giga": 0}
    rep_examples = []
    del SWAPS[:]
    for i, seg in enumerate(segments):
        a, b = float(seg["a"]), float(seg["b"])
        chunk = os.path.join(temp, f"seg_{i:03d}.wav")
        off = cut_chunk(voice, a, b, chunk, voice_dur)
        tw = transcribe_wav(model, chunk)
        # времена Whisper -> время исходника; берём то, что пересекается с [a, b)
        wh = [{"w": x["w"], "start": x["start"] + off, "end": x["end"] + off}
              for x in tw]
        wh = [x for x in wh if x["start"] < b and x["end"] > a
              and norm_word(x["w"])]
        gw = [g for g in giga if a - 1e-9 <= float(g["start"]) < b]

        if not wh:
            # Whisper по сегменту пусто — слова GigaAM остаются как есть
            res = [_pair(g["w"], g["w"], g["start"], g["end"], "giga") for g in gw]
            reps = []
            for x in res:
                stats["giga"] += 1
        else:
            res, reps = align(gw, wh, a, b)
            for x in res:
                stats[x["src"]] += 1
        all_words.extend(res)
        per_seg.append({"i": i, "a": a, "b": b, "giga": gw, "res": res})
        rep_examples.extend((i, s) for s in reps if s[0] != s[1])
        log(f"  сегмент {i:2d} [{a:.3f}..{b:.3f})  GigaAM {len(gw):2d} -> "
            f"{len(res):2d}  (eq {sum(1 for x in res if x['src']=='eq')}, "
            f"rep {sum(1 for x in res if x['src']=='rep')}, "
            f"ins {sum(1 for x in res if x['src']=='ins')}, "
            f"del {sum(1 for x in res if x['src']=='del')}, "
            f"giga {sum(1 for x in res if x['src']=='giga')})")
        try:
            os.remove(chunk)
        except OSError:
            pass
    dt = time.time() - t0

    # --- порядок по времени + снятие дублей (сегменты EDL могут повторяться)
    all_words.sort(key=lambda x: (x["start"], x["end"]))
    words, seen = [], []
    for x in all_words:
        nx = norm_word(x["w"])
        dup = any(nx == k[0] and abs(x["start"] - k[1]) <= 0.25 for k in seen)
        if dup:
            continue
        seen.append((nx, x["start"]))
        words.append(x)

    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(words, fh, ensure_ascii=False, indent=1)

    # --- статистика — по тому, что реально записано (после снятия дублей)
    fin = {k: sum(1 for x in words if x["src"] == k) for k in stats}

    # --- .txt: по сегментам строка GigaAM и строка итоговая
    lines = [f"# {name} — текст Whisper ({MODEL_SIZE}), время GigaAM; src={src}",
             f"# слов: {len(words)}  ({', '.join(f'{k} {v}' for k, v in fin.items())})"]
    for s in per_seg:
        lines.append("")
        n = {k: sum(1 for x in s["res"] if x["src"] == k) for k in stats}
        lines.append(f"# сегмент {s['i']}  [{s['a']:.3f}..{s['b']:.3f})  "
                     + ", ".join(f"{k} {v}" for k, v in n.items() if v))
        lines.append("GIGA: " + " ".join(g["w"] for g in s["giga"]))
        lines.append("ИТОГ: " + " ".join(x["w"] for x in s["res"]))
    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    log(f"  Whisper: {dt:.1f} с на {len(segments)} сегментов "
        f"({len(segments) / dt:.1f} сегм/с)" if dt > 0 else "")
    log(f"  слов: {len(all_words)} -> {len(words)} (снято дублей: "
        f"{len(all_words) - len(words)})   "
        + "  ".join(f"{k}={v}" for k, v in fin.items()))
    for i, (was, now) in rep_examples:
        log(f"    rep сегмент {i}: {was!r} -> {now!r}")
    if SWAPS:
        log(f"  мат, «почищенный» Whisper (в `w` оставлено слово GigaAM): "
            f"{len(SWAPS)}")
        for w0, g0 in SWAPS:
            log(f"    Whisper {w0!r} -> GigaAM {g0!r}")
    log(f"  записано: {out_json}")
    log(f"            {out_txt}")

    try:
        os.rmdir(temp)
    except OSError:
        pass
    return {"name": name, "skipped": False, "words": len(words), "stats": fin,
            "whisper_time": dt, "segments": len(segments), "rep": rep_examples,
            "swaps": list(SWAPS)}


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description="Слова: Whisper (текст) + GigaAM (время)")
    ap.add_argument("edl", nargs="+", help="монтажный лист (JSON)")
    ap.add_argument("--force", action="store_true",
                    help="переписать .words.json, даже если он новее EDL")
    args = ap.parse_args()

    t_all = time.time()
    total = {"eq": 0, "rep": 0, "ins": 0, "del": 0, "giga": 0}
    t_whisper = 0.0
    ok = True
    for edl_path in args.edl:
        try:
            st = process_edl(edl_path, args.force)
        except Exception as exc:                      # noqa: BLE001
            log(f"ОШИБКА на {edl_path}: {exc}")
            ok = False
            continue
        if st.get("skipped"):
            continue
        for k in total:
            total[k] += st["stats"][k]
        t_whisper += st["whisper_time"]

    log("=" * 78)
    log("ИТОГО: " + "  ".join(f"{k}={v}" for k, v in total.items()))
    log(f"Whisper (без загрузки модели): {t_whisper:.1f} с; "
        f"загрузка модели: {MODEL_LOAD_TIME:.1f} с; "
        f"всего прогон: {time.time() - t_all:.1f} с")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
