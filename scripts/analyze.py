# -*- coding: utf-8 -*-
"""Расшифровка стрима GigaAM v3_ctc + всплески громкости.

# Adapted from Reelsi (https://github.com/mxmlab/reelsi), AGPL-3.0, reelsi/core/gigaam_cut/asr.py

Использование:
    python analyze.py <путь к видео> [--force]

Кладёт в <analysis_dir>/<имя видео без расширения>/:
    audio.wav       — моно 16 кГц pcm_s16le (первая аудиодорожка, ffmpeg)
    words.json      — [{"w","start","end"}] (секунды, float, 3 знака)
    transcript.txt  — фразы с таймкодами "[HH:MM:SS] текст"
    loudness.json   — RMS в дБ по окнам 0.5 с + найденные пики
    peaks.txt       — "[HH:MM:SS] +NN dB (длительность N с)"

`analysis_dir` — из конфига (`<stream_root>/_analysis` по умолчанию, см. config.py);
перебить можно `--out-dir`.

Готовый файл-результат = этап пропускается (кэш); --force пересчитывает всё.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG  # noqa: E402

SR = 16000                     # GigaAM ждёт 16 кГц
FFMPEG = CONFIG.ffmpeg
FFPROBE = CONFIG.ffprobe

WIN_SEC = 18.0                 # окно распознавания, сек
SEARCH_SEC = 6.0               # где искать тихий шов в конце окна, сек
PAUSE_SEC = 0.8                # пауза между словами -> новая фраза
PHRASE_SEC = 15.0              # предельная длина фразы, сек
RMS_WIN_SEC = 0.5              # окно RMS, сек
PEAK_DB = 12.0                 # пик = выше медианы на столько дБ
MERGE_SEC = 5.0                # склейка соседних пиков ближе этого, сек

BASE_DIR = CONFIG.analysis_dir


def emit(msg: str, **kw: Any) -> None:
    """Замена эмиттера библиотеки: тот же вызов, но печать в stdout (эмит -> print)."""
    print(msg.format(**kw) if kw else msg, flush=True)


def _free_torch() -> None:
    """Освободить VRAM (копия _free_torch из заимствованного модуля, без его исключения)."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass  # torch/GPU недоступны — чистить нечего


# --------------------------------------------------------------------------- #
# Скопировано из заимствованного модуля (см. строку атрибуции в шапке файла)
# --------------------------------------------------------------------------- #
def _word(w: Any, t0: float = 0.0) -> dict[str, Any]:
    """gigaam.Word -> наш словарь [{w,start,end}] (t0 — сдвиг окна, сек).
    `prob` gigaam не отдаёт; если появится — подхватим (её показывает self-check)."""
    d = {"w": w.text, "start": round(float(w.start) + t0, 3),
         "end": round(float(w.end) + t0, 3)}
    p = getattr(w, "prob", None)
    if p is not None:
        d["prob"] = round(float(p), 3)
    return d


def _quiet_cut(x: Any, lo: int, hi: int, sr: int, frame: float = 0.05) -> int:
    """Самая тихая точка (центр самого тихого 50мс-кадра) в x[lo:hi]."""
    f = max(1, int(frame * sr))
    seg = np.abs(x[lo:hi].astype(np.float32))
    nf = len(seg) // f
    if nf < 1:
        return hi
    rms = seg[:nf * f].reshape(nf, f).mean(axis=1)
    return lo + int(np.argmin(rms)) * f + f // 2


def _transcribe_words_manual(
    model: Any, wav_path: str, emit: Any = emit, win: float = WIN_SEC,
    search: float = SEARCH_SEC, sr: int = SR, tmp_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Фолбэк без pyannote: окна ~win сек, но шов кладём в самую тихую точку
    последних `search` сек окна — рез между окнами не попадает в слово.
    Каждое окно transcribe(word_timestamps=True), тайминги + начало окна.

    Окно 18с и torch.cuda.empty_cache() ПОСЛЕ каждого окна — раньше этот фолбэк
    ронял процесс с кодом 3221225477 (access violation): на Windows WDDM пик VRAM
    в тихом пооконном GPU-цикле переполнял память без ловимого OOM и убивал драйвер.
    Меньше окно + сброс кэша держат запас; прогресс печатаем — крах локализуется."""
    import os
    a, _sr = sf.read(wav_path, dtype="int16")
    if _sr != sr:
        emit("  ! wav не 16кГц — GigaAM ожидает 16кГц, возможны артефакты", flush=True)
    if a.ndim > 1:
        a = a.mean(1).astype("int16")
    n = len(a)
    step = int(win * sr)
    words: list[dict[str, Any]]
    pos: int
    nwin: int
    words, pos, nwin = [], 0, 0
    est = max(1, int(n / step) + 1)              # грубая оценка числа окон для прогресса
    if tmp_dir is None:
        tmp_dir = tempfile.gettempdir()
    while pos < n:
        if n - pos > step:
            s1 = _quiet_cut(a, pos + step - int(search * sr), pos + step, sr)
        else:
            s1 = n
        if s1 - pos < int(0.5 * sr):
            break
        tmp = os.path.join(tmp_dir, "_gc_win_%d_%d.wav" % (os.getpid(), pos))
        try:
            sf.write(tmp, a[pos:s1], sr, subtype="PCM_16")
            r = model.transcribe(tmp, word_timestamps=True)
            t0 = pos / sr
            for w in (r.words or []):
                if w.text:
                    words.append(_word(w, t0))
            nwin += 1
        except Exception as ex:
            emit("  окно {sec}s не расшифровалось: {err}", sec=pos // sr, err=str(ex), flush=True)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass  # временное окно уже убрано
            _free_torch()                        # держим VRAM в узде (Windows WDDM)
        emit("  окно {cur}/{est} ({words} слов)…", cur=nwin, est=est, words=len(words))
        pos = s1
    emit("  longform-окна (тихие швы): {chunks} кусков, {words} слов",
         chunks=nwin, words=len(words), flush=True)
    return words


# --------------------------------------------------------------------------- #
# Этапы
# --------------------------------------------------------------------------- #
def video_duration(video: str) -> float:
    """Длительность видео в секундах (ffprobe), 0.0 если не удалось."""
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", video],
            capture_output=True, text=True, check=True).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


def extract_audio(video: str, wav_path: str) -> None:
    """Этап 1: первая аудиодорожка -> моно 16 кГц pcm_s16le."""
    dur = video_duration(video)
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", video,
           "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SR),
           "-c:a", "pcm_s16le", "-progress", "pipe:1", "-nostats", wav_path]
    emit("ffmpeg: извлекаю аудио ({dur:.0f} с)…", dur=dur)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    last = 0.0
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line.startswith("out_time="):
            continue
        try:                       # out_time=HH:MM:SS.ffffff — без двусмысленности мс/мкс
            h, m, s = line.split("=", 1)[1].split(":")
            sec = int(h) * 3600 + int(m) * 60 + float(s)
        except ValueError:
            continue
        if dur and sec - last >= 30:
            last = sec
            emit("  аудио {cur:.0f}/{dur:.0f} с…", cur=sec, dur=dur)
    proc.wait()
    err = (proc.stderr.read() if proc.stderr else "") or ""
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg упал (%d): %s" % (proc.returncode, err.strip()[:500]))
    emit("  audio.wav готов: {mb:.1f} МБ", mb=os.path.getsize(wav_path) / 1048576)


def transcribe(wav_path: str, out_dir: str) -> list[dict[str, Any]]:
    """Этап 2: GigaAM v3_ctc окнами ~18 с -> слова с таймингами (секунды)."""
    import gigaam
    emit("GigaAM: загрузка v3_ctc (word_timestamps, окна {win}с)…", win=WIN_SEC)
    model = gigaam.load_model("v3_ctc")
    try:
        words = _transcribe_words_manual(model, wav_path, emit=emit, tmp_dir=out_dir)
    finally:
        del model
        _free_torch()
    words = [w for w in words if w["w"].strip()]
    words.sort(key=lambda w: w["start"])
    emit("GigaAM: {count} слов (родные тайминги)", count=len(words))
    return words


def build_transcript(words: list[dict[str, Any]]) -> str:
    """Этап 3: слова -> фразы. Новая фраза при паузе > 0.8 с или длине > 15 с."""
    lines: list[str] = []
    cur: list[dict[str, Any]] = []
    start = end = 0.0
    for w in words:
        if cur and (w["start"] - end > PAUSE_SEC or w["end"] - start > PHRASE_SEC):
            lines.append("%s %s" % (tc(start), " ".join(x["w"] for x in cur)))
            cur = []
        if not cur:
            start = w["start"]
        cur.append(w)
        end = w["end"]
    if cur:
        lines.append("%s %s" % (tc(start), " ".join(x["w"] for x in cur)))
    return "\n".join(lines) + ("\n" if lines else "")


def tc(sec: float) -> str:
    """Секунды -> [HH:MM:SS]."""
    s = int(sec)
    return "[%02d:%02d:%02d]" % (s // 3600, (s % 3600) // 60, s % 60)


def loudness(wav_path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Этап 4: RMS в дБ по окнам 0.5 с, пики выше медианы на >= 12 дБ.

    Файл читается блоками по ~1000 с (2 ч аудио целиком в float64 не влезает),
    окна 0.5 с не рвутся: блок всегда кратен окну."""
    with sf.SoundFile(wav_path) as f:
        sr = f.samplerate
        total = len(f)
        win = max(1, int(RMS_WIN_SEC * sr))
        block = win * int(1000 / RMS_WIN_SEC)
        dbs: list[Any] = []
        carry = np.zeros(0, dtype="float32")
        done = 0
        while True:
            data = f.read(block, dtype="float32", always_2d=True)
            if not len(data):
                break
            mono = np.concatenate((carry, data.mean(axis=1))) if len(carry) else data.mean(axis=1)
            n = len(mono) // win
            if n:
                frames = mono[:n * win].reshape(n, win)
                rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
                dbs.append(rms)
            carry = mono[n * win:]
            done += len(data)
            emit("  громкость {cur:.0f}/{dur:.0f} с…", cur=done / sr, dur=total / sr)
    db = 20.0 * np.log10(np.maximum(np.concatenate(dbs), 1e-10))
    nwin = len(db)
    if nwin < 1:
        raise RuntimeError("аудио короче одного окна RMS")
    median = float(np.median(db))
    thr = median + PEAK_DB
    peak_idx = np.nonzero(db >= thr)[0]
    emit("громкость: {nwin} окон по {w} с, медиана {med:.1f} дБ, порог {thr:.1f} дБ, "
         "пиковых окон {p}", nwin=nwin, w=RMS_WIN_SEC, med=median, thr=thr, p=len(peak_idx))

    peaks: list[dict[str, Any]] = []
    i = 0
    while i < len(peak_idx):
        j = i
        while j + 1 < len(peak_idx) and (peak_idx[j + 1] - peak_idx[j]) * RMS_WIN_SEC < MERGE_SEC:
            j += 1
        group = peak_idx[i:j + 1]
        start = float(group[0]) * RMS_WIN_SEC
        end = float(group[-1] + 1) * RMS_WIN_SEC
        top = float(db[group].max())
        peaks.append({"start": round(start, 3), "end": round(end, 3),
                      "peak_db": round(top, 2), "delta_db": round(top - median, 2),
                      "duration_sec": round(end - start, 3)})
        i = j + 1

    doc = {
        "window_sec": RMS_WIN_SEC,
        "duration_sec": round(total / sr, 3),
        "median_db": round(median, 2),
        "threshold_db": round(thr, 2),
        "threshold_offset_db": PEAK_DB,
        "merge_gap_sec": MERGE_SEC,
        "peak_windows": int(len(peak_idx)),
        "windows": [{"start": round(k * RMS_WIN_SEC, 3), "db": round(float(v), 2)}
                    for k, v in enumerate(db)],
        "peaks": peaks,
    }
    return doc, peaks


def peaks_txt(peaks: list[dict[str, Any]]) -> str:
    """peaks.txt: [HH:MM:SS] +NN dB (длительность N с), по времени."""
    lines = ["%s +%.1f dB (длительность %.1f с)" % (tc(p["start"]), p["delta_db"],
                                                    p["duration_sec"])
             for p in sorted(peaks, key=lambda p: p["start"])]
    return "\n".join(lines) + ("\n" if lines else "")


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
        f.write("\n")


def stage(path: str, force: bool, label: str) -> bool:
    """True — этап надо считать (файла нет или --force)."""
    if os.path.exists(path) and not force:
        emit("= {label}: уже есть {name}, пропускаю (--force чтобы пересчитать)",
             label=label, name=os.path.basename(path))
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="GigaAM-расшифровка + всплески громкости")
    ap.add_argument("video", help="путь к видеофайлу")
    ap.add_argument("--force", action="store_true", help="пересчитать всё заново")
    ap.add_argument("--out-dir", default=None,
                    help="куда класть <имя>/ (по умолчанию analysis_dir из конфига)")
    args = ap.parse_args(argv)

    video = os.path.abspath(args.video)
    if not os.path.isfile(video):
        emit("нет такого файла: {p}", p=video)
        return 2
    stem = os.path.splitext(os.path.basename(video))[0]
    out_dir = os.path.join(args.out_dir or BASE_DIR, stem)
    os.makedirs(out_dir, exist_ok=True)
    emit("== {name} -> {out}", name=os.path.basename(video), out=out_dir)

    t_all = time.time()
    wav_path = os.path.join(out_dir, "audio.wav")
    words_path = os.path.join(out_dir, "words.json")
    txt_path = os.path.join(out_dir, "transcript.txt")
    loud_path = os.path.join(out_dir, "loudness.json")
    peaks_path = os.path.join(out_dir, "peaks.txt")

    # 1. аудио
    if stage(wav_path, args.force, "audio"):
        t = time.time()
        extract_audio(video, wav_path)
        emit("  [audio] {sec:.1f} с", sec=time.time() - t)

    # 2. слова
    if stage(words_path, args.force, "words"):
        t = time.time()
        words = transcribe(wav_path, out_dir)
        write_json(words_path, words)
        emit("  [words] {sec:.1f} с", sec=time.time() - t)

    # 3. фразы
    if stage(txt_path, args.force, "transcript"):
        with open(words_path, encoding="utf-8") as f:
            words = json.load(f)
        text = build_transcript(words)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        emit("transcript.txt: {n} фраз", n=text.count("\n"))

    # 4. громкость и пики
    if stage(loud_path, args.force, "loudness") or stage(peaks_path, args.force, "peaks"):
        t = time.time()
        doc, peaks = loudness(wav_path)
        write_json(loud_path, doc)
        with open(peaks_path, "w", encoding="utf-8") as f:
            f.write(peaks_txt(peaks))
        emit("  [loudness] {sec:.1f} с", sec=time.time() - t)

    with open(words_path, encoding="utf-8") as f:
        n_words = len(json.load(f))
    with open(peaks_path, encoding="utf-8") as f:
        n_peaks = sum(1 for line in f if line.strip())
    emit("== готово: {w} слов, {p} пиков, всего {sec:.1f} с",
         w=n_words, p=n_peaks, sec=time.time() - t_all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
