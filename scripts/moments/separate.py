# -*- coding: utf-8 -*-
"""Шаг 1: голос из записи стрима (audio-separator, vocals_mel_band_roformer).

    python separate.py <имя> [--force]

<имя> — имя исходника без расширения (game1, game2, …), видео берётся единым
поиском `scripts/sources.py`. Результат — <analysis_dir>/<имя>/voice.wav
(моно 16 кГц pcm_s16le); `--out-root` переносит запись в другую папку.

Как в задании:
  1. ffmpeg: первая аудиодорожка видео -> куски по 600 с, стерео 44.1 кГц WAV во временную папку;
  2. каждый кусок последовательно через audio-separator;
  3. результаты склеить по порядку и свести в моно 16 кГц;
  4. длительность voice.wav обязана совпасть с audio.wav ±0.1 с;
  5. кэш: voice.wav есть и длительность сходится -> пропуск (--force пересчитывает).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import soundfile as sf

BASE = Path(__file__).resolve().parent                  # scripts/moments
REPO = BASE.parent.parent                               # корень репозитория
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from config import CONFIG                                # noqa: E402
from sources import find_source                          # noqa: E402  (единый поиск исходника)

ANALYSIS = Path(CONFIG.analysis_dir)

SEP_EXE = CONFIG.separator_exe
SEP_MODEL_DIR = CONFIG.separator_model_dir
SEP_MODEL = CONFIG.separator_model

CHUNK_SEC = 600.0
CHUNK_SR = 44100
OUT_SR = 16000
TOL_SEC = 0.1
VIDEO_EXT = (".mov", ".mp4", ".mkv", ".webm")


def log(msg: str) -> None:
    print(msg, flush=True)


def duration(path: Path) -> float:
    """Длительность аудиофайла в секундах (soundfile)."""
    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def video_path(name: str) -> Path:
    try:
        return Path(find_source(name))
    except FileNotFoundError as ex:
        raise SystemExit(str(ex))


def run(cmd: list[str], quiet: bool = False, env: dict | None = None) -> str:
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, encoding="utf-8", errors="replace", env=env)
    out = r.stdout or ""
    if r.returncode != 0:
        raise RuntimeError("команда упала (%d):\n%s\n%s" % (r.returncode, " ".join(cmd), out[-2000:]))
    if not quiet and out.strip():
        log("    " + out.strip().splitlines()[-1][:200])
    return out


def sep_env() -> dict:
    """Окружение для audio-separator.

    numba (через librosa) по умолчанию кладёт свой дисковый кэш рядом с пакетом в
    site-packages; если тот недоступен на запись, `ensure_cache_path` зацикливается и
    загрузка модели висит бесконечно. Поэтому NUMBA_CACHE_DIR уводим в рабочую папку.
    """
    env = dict(os.environ)
    cache = Path(env.setdefault("NUMBA_CACHE_DIR", str(Path(CONFIG.temp_dir) / "numba")))
    cache.mkdir(parents=True, exist_ok=True)
    return env


def split_video(video: Path, tmp: Path) -> list[Path]:
    """Первая аудиодорожка -> куски по CHUNK_SEC, стерео 44.1 кГц WAV."""
    t0 = time.time()
    pattern = str(tmp / "chunk_%03d.wav")
    run([CONFIG.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(video), "-map", "0:a:0", "-vn",
         "-ac", "2", "-ar", str(CHUNK_SR), "-c:a", "pcm_s16le",
         "-f", "segment", "-segment_time", str(int(CHUNK_SEC)), "-reset_timestamps", "1",
         pattern], quiet=True)
    chunks = sorted(tmp.glob("chunk_*.wav"))
    if not chunks:
        raise RuntimeError("ffmpeg не нарезал ни одного куска")
    log("  ffmpeg: %d кусок(ов) по %d с за %.0f с" % (len(chunks), int(CHUNK_SEC), time.time() - t0))
    return chunks


def separate_chunk(chunk: Path, out_dir: Path) -> Path:
    """Один кусок через audio-separator; возвращает единственный WAV из out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [SEP_EXE, str(chunk), "-m", SEP_MODEL, "--model_file_dir", SEP_MODEL_DIR,
           "--output_dir", str(out_dir), "--output_format", "WAV", "--single_stem", "Vocals"]
    run(cmd, quiet=True, env=sep_env())
    wavs = sorted(out_dir.glob("*.wav"))
    if len(wavs) != 1:
        raise RuntimeError("в %s ожидался один WAV, найдено %d: %s"
                           % (out_dir, len(wavs), [w.name for w in wavs]))
    return wavs[0]


def concat_mono(vocals: list[Path], tmp: Path, dst: Path) -> None:
    """Склейка по порядку + сведение в моно 16 кГц pcm_s16le."""
    lst = tmp / "concat.txt"
    with open(lst, "w", encoding="utf-8") as f:
        for p in vocals:
            f.write("file '%s'\n" % str(p).replace("\\", "/").replace("'", "'\\''"))
    run([CONFIG.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "concat", "-safe", "0", "-i", str(lst),
         "-ac", "1", "-ar", str(OUT_SR), "-c:a", "pcm_s16le", str(dst)], quiet=True)


def separate(name: str, force: bool, out_root: Path = ANALYSIS) -> int:
    src_dir = ANALYSIS / name               # audio.wav — только чтение
    out_dir = out_root / name               # voice.wav — запись
    audio = src_dir / "audio.wav"
    voice = out_dir / "voice.wav"
    if not audio.is_file():
        log("! нет %s — нечего сверять по длительности" % audio)
        return 2

    d_audio = duration(audio)
    if voice.is_file() and not force:
        d_voice = duration(voice)
        if abs(d_voice - d_audio) <= TOL_SEC:
            log("= %s: voice.wav уже есть (%.3f с, audio.wav %.3f с) — пропуск (--force чтобы пересчитать)"
                % (name, d_voice, d_audio))
            return 0
        log("! %s: voice.wav %.3f с не сходится с audio.wav %.3f с — пересчитываю"
            % (name, d_voice, d_audio))

    video = video_path(name)
    log("== %s: %s (%.0f с аудио)" % (name, video.name, d_audio))
    t_all = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="sep_%s_" % name))
    try:
        chunks = split_video(video, tmp)
        vocals: list[Path] = []
        for i, chunk in enumerate(chunks, 1):
            t0 = time.time()
            wav = separate_chunk(chunk, tmp / ("out_%03d" % i))
            vocals.append(wav)
            log("  кусок %d/%d: %.3f с аудио -> %.3f с голоса (%s, %.0f с счёта)"
                % (i, len(chunks), duration(chunk), duration(wav), wav.name, time.time() - t0))
        raw = tmp / "voice_cat.wav"
        concat_mono(vocals, tmp, raw)
        d_raw = duration(raw)
        if abs(d_raw - d_audio) > TOL_SEC:
            raise RuntimeError("склейка %.3f с не сходится с audio.wav %.3f с" % (d_raw, d_audio))
        out_dir.mkdir(parents=True, exist_ok=True)
        os.replace(raw, voice)          # temp и рабочая папка на одном томе
        d_voice = duration(voice)
        if abs(d_voice - d_audio) > TOL_SEC:
            raise RuntimeError("voice.wav %.3f с не сходится с audio.wav %.3f с" % (d_voice, d_audio))
        log("[separate] %s: %.1f с, voice.wav %.3f с (audio.wav %.3f с)"
            % (name, time.time() - t_all, d_voice, d_audio))
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Отделение голоса из видео (audio-separator)")
    ap.add_argument("name", help="имя исходника без расширения")
    ap.add_argument("--force", action="store_true", help="пересчитать даже при готовом voice.wav")
    ap.add_argument("--out-root", default=str(ANALYSIS),
                    help="куда класть <имя>/voice.wav (по умолчанию analysis_dir)")
    args = ap.parse_args(argv)
    try:
        return separate(args.name, args.force, Path(args.out_root))
    except Exception as ex:
        log("! %s: %s" % (args.name, ex))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
