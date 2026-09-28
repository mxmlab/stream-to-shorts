# -*- coding: utf-8 -*-
"""Шаг 2: посекундные признаки -> <analysis_dir>/<имя>/features.csv

    python features.py <имя> [--force] [--batch N] [--device cuda|cpu]

Сетка t = 0,1,2,… (целые секунды). Окно моделей — 3 с: [t-1, t+2), края дополняются нулями.
Столбцы: t, raw_db, voice_db, arousal, dominance, valence, emo_<id2name…>, laugh, scream, words, markers.

Модели:
  * audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim — EmotionModel с RegressionHead
    «ровно как в карточке» (порядок выходов arousal, dominance, valence);
  * gigaam "emo" — батчевый путь preprocessor -> encoder -> avg_pool -> head -> softmax
    (как внутри GigaAMEmo.get_probs), имена классов из model.id2name;
  * MIT/ast-finetuned-audioset-10-10-0.4593 — сигмоида логитов, laugh/scream = max по меткам.

Кэш: features.csv есть и число строк сходится с длительностью (±1) -> пропуск (--force пересчитывает).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

BASE = Path(__file__).resolve().parent                  # scripts/moments
REPO = BASE.parent.parent                               # корень репозитория
sys.path.insert(0, str(REPO))
from config import CONFIG                               # noqa: E402

ANALYSIS = Path(CONFIG.analysis_dir)

SR = 16000
WIN_SEC = 3.0                 # окно модели, с
WIN = int(SR * WIN_SEC)
MARKERS = ["а", "аа", "ааа", "ай", "ой", "блин", "бля", "блять", "сука", "пиздец", "нахуй", "хуй",
           "ебать", "господи", "мама", "мамочки", "что", "чего", "фу", "стоп", "нет", "помогите",
           "капец", "жесть", "страшно", "ахаха", "хаха", "хах"]

MODEL_DIM = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
MODEL_AST = "MIT/ast-finetuned-audioset-10-10-0.4593"
LAUGH_LABELS = ["Laughter", "Giggle", "Belly laugh", "Chuckle, chortle", "Snicker", "Baby laughter"]
SCREAM_LABELS = ["Screaming", "Shout", "Yell", "Children shouting", "Gasp"]

RMS_FLOOR = 1e-10


def log(msg: str) -> None:
    print(msg, flush=True)


def norm_word(w: str) -> str:
    """Сравнение маркеров: нижний регистр, ё -> е."""
    return w.strip().lower().replace("ё", "е")


# --------------------------------------------------------------------------- #
# кэши моделей
# --------------------------------------------------------------------------- #
def _writable(d: Path) -> bool:
    try:
        d.mkdir(parents=True, exist_ok=True)
        p = d / ".write_probe"
        p.write_text("x")
        p.unlink()
        return True
    except OSError:
        return False


def ensure_hf_cache() -> None:
    """HF-кэш по умолчанию — пользовательский; если он не доступен на запись
    (сессия в песочнице), уводим HF_HOME в _cache рабочей папки. Вызывать до импорта transformers."""
    if os.environ.get("HF_HOME"):
        return
    default = Path.home() / ".cache" / "huggingface"
    if _writable(default / "hub"):
        return
    tmp = BASE / "_cache" / "hf_home"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(tmp)
    log("! кэш HF %s недоступен на запись — HF_HOME=%s" % (default, tmp))


def gigaam_cache_dir() -> str | None:
    """Каталог кэша gigaam, если пользовательский недоступен на запись (иначе None = по умолчанию)."""
    default = Path(os.path.expanduser("~/.cache/gigaam"))
    if _writable(default):
        return None
    tmp = BASE / "_cache" / "gigaam"
    tmp.mkdir(parents=True, exist_ok=True)
    log("! кэш gigaam %s недоступен на запись — использую %s" % (default, tmp))
    return str(tmp)


# --------------------------------------------------------------------------- #
# аудио
# --------------------------------------------------------------------------- #
def rms_db_per_sec(path: Path, n_sec: int) -> np.ndarray:
    """RMS в дБ по секундам [t, t+1) (блоками, чтобы не держать файл в памяти)."""
    sumsq = np.zeros(n_sec, dtype=np.float64)
    cnt = np.zeros(n_sec, dtype=np.int64)
    with sf.SoundFile(str(path)) as f:
        sr = f.samplerate
        if sr != SR:
            raise RuntimeError("%s: %d Гц вместо %d" % (path, sr, SR))
        block = sr * 600
        pos = 0
        while True:
            data = f.read(block, dtype="float32", always_2d=True)
            if not len(data):
                break
            mono = data.mean(axis=1).astype(np.float64)
            sec = (pos + np.arange(len(mono), dtype=np.int64)) // sr
            keep = sec < n_sec
            if not keep.all():
                mono, sec = mono[keep], sec[keep]
            sumsq += np.bincount(sec, weights=mono * mono, minlength=n_sec)[:n_sec]
            cnt += np.bincount(sec, minlength=n_sec)[:n_sec]
            pos += len(data)
    if (cnt == 0).any():
        log("  ! пустых секунд: %d (там будет NaN)" % int((cnt == 0).sum()))
    rms = np.divide(sumsq, np.maximum(cnt, 1), out=np.full(n_sec, np.nan), where=cnt > 0)
    return 20.0 * np.log10(np.maximum(np.sqrt(rms), RMS_FLOOR))


def read_int16(path: Path) -> np.ndarray:
    """Моно-сигнал как int16 (в 2 раза компактнее float32)."""
    x, sr = sf.read(str(path), dtype="int16", always_2d=False)
    if sr != SR:
        raise RuntimeError("%s: %d Гц вместо %d" % (path, sr, SR))
    if x.ndim > 1:                     # моно-файл, но на всякий случай
        x = x[:, 0]
    return x


def windows(x: np.ndarray, t_list: list[int]) -> np.ndarray:
    """Батч окон [t-1, t+2) (края дополняются нулями), float32 [-1, 1]."""
    out = np.zeros((len(t_list), WIN), dtype=np.float32)
    n = len(x)
    for i, t in enumerate(t_list):
        s = (t - 1) * SR
        a, b = max(0, s), min(n, s + WIN)
        if b > a:
            out[i, a - s:b - s] = x[a:b].astype(np.float32) / 32768.0
    return out


# --------------------------------------------------------------------------- #
# модели
# --------------------------------------------------------------------------- #
def load_models(device: str):
    """audeering (EmotionModel как в карточке) + AST + gigaam emo."""
    import torch
    import torch.nn as nn
    from transformers import (ASTFeatureExtractor, ASTForAudioClassification,
                              Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2PreTrainedModel)

    class RegressionHead(nn.Module):
        """Голова из карточки audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim."""

        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, features, **kwargs):
            x = self.dropout(features)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            return self.out_proj(x)

    class EmotionModel(Wav2Vec2PreTrainedModel):
        """Класс модели из карточки: wav2vec2 + RegressionHead, выход (hidden, logits)."""

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.init_weights()

        def forward(self, input_values):
            hidden_states = self.wav2vec2(input_values)
            hidden_states = torch.mean(hidden_states[0], dim=1)
            return hidden_states, self.classifier(hidden_states)

    t0 = time.time()
    dim_ext = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_DIM)
    dim = EmotionModel.from_pretrained(MODEL_DIM).to(device).eval()
    log("  audeering: загружен за %.0f с (порядок выходов arousal, dominance, valence)" % (time.time() - t0))

    t0 = time.time()
    ast_ext = ASTFeatureExtractor.from_pretrained(MODEL_AST)
    ast = ASTForAudioClassification.from_pretrained(MODEL_AST).to(device).eval()
    id2label = {int(k): v for k, v in ast.config.id2label.items()}

    def label_ids(names: list[str], kind: str) -> list[int]:
        idx: list[int] = []
        for n in names:
            hits = sorted(i for i, v in id2label.items() if v == n)
            if not hits:
                log("  ! AST: метка %r не найдена — пропускаю" % n)
                continue
            idx.extend(hits)
        log("  AST %s: %d меток %s" % (kind, len(idx), [id2label[i] for i in idx]))
        return idx

    laugh_idx = label_ids(LAUGH_LABELS, "laugh")
    scream_idx = label_ids(SCREAM_LABELS, "scream")
    if not laugh_idx:
        raise RuntimeError("у AST не нашлось ни одной метки смеха")
    if not scream_idx:
        raise RuntimeError("у AST не нашлось ни одной метки крика")
    log("  AST: загружен за %.0f с" % (time.time() - t0))

    t0 = time.time()
    import gigaam
    cache = gigaam_cache_dir()
    if cache:
        gigaam._CACHE_DIR = cache          # тот же путь загрузки, что и gigaam.load_model("emo")
    emo = gigaam.load_model("emo")
    id2name = emo.id2name                  # у emo это список имён классов (индекс -> имя)
    if isinstance(id2name, dict):
        mapping = {int(k): str(v) for k, v in id2name.items()}
    else:
        mapping = {i: str(v) for i, v in enumerate(id2name)}
    emo_cols = ["emo_" + norm_word(mapping[i]).replace(" ", "_") for i in sorted(mapping)]
    log("  gigaam emo: id2name=%s -> столбцы %s (за %.0f с)"
        % ({i: mapping[i] for i in sorted(mapping)}, emo_cols, time.time() - t0))

    return {"dim": dim, "dim_ext": dim_ext, "ast": ast, "ast_ext": ast_ext,
            "laugh_idx": np.array(laugh_idx), "scream_idx": np.array(scream_idx),
            "emo": emo, "emo_cols": emo_cols, "device": device}


def run_batch(m, wav: np.ndarray):
    """wav float32 [B, WIN] -> arousal, dominance, valence, laugh, scream, emo_probs [B, C]."""
    import torch

    dev = m["device"]

    # audeering: нормализация тем же feature extractor'ом, что в карточке
    inp = m["dim_ext"](list(wav), sampling_rate=SR, return_tensors="pt")["input_values"].to(dev)
    with torch.inference_mode(), torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")):
        _, logits = m["dim"](inp)
    dim = logits.float().cpu().numpy()

    # AST: сигмоида логитов
    a_in = {k: v.to(dev) for k, v in m["ast_ext"](list(wav), sampling_rate=SR, return_tensors="pt").items()}
    with torch.inference_mode():
        a_logits = m["ast"](**a_in).logits
    probs = torch.sigmoid(a_logits.float()).cpu().numpy()
    laugh = probs[:, m["laugh_idx"]].max(axis=1)
    scream = probs[:, m["scream_idx"]].max(axis=1)

    # gigaam emo: тот же путь, что в GigaAMEmo.get_probs, но батчем
    emo = m["emo"]
    w = torch.from_numpy(wav).to(dev).to(next(emo.parameters()).dtype)
    lens = torch.full([wav.shape[0]], wav.shape[1], device=dev)
    with torch.inference_mode():
        encoded, _ = emo.forward(w, lens)
        pooled = torch.nn.functional.avg_pool1d(encoded, kernel_size=encoded.shape[-1]).squeeze(-1)
        e_probs = torch.nn.functional.softmax(emo.head(pooled), dim=-1).float().cpu().numpy()

    return dim[:, 0], dim[:, 1], dim[:, 2], laugh, scream, e_probs


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def compute(name: str, force: bool, batch: int, device: str, out_root: Path = ANALYSIS) -> int:
    out_dir = out_root / name
    src_dir = ANALYSIS / name
    audio, voice = src_dir / "audio.wav", out_dir / "voice.wav"
    csv_path = out_dir / "features.csv"
    if not voice.is_file():
        log("! %s: нет voice.wav — сначала separate.py" % name)
        return 2
    if not audio.is_file():
        log("! %s: нет audio.wav" % name)
        return 2

    dur = sf.info(str(audio)).frames / float(SR)
    n_sec = int(math.ceil(dur - 1e-9))
    if csv_path.is_file() and not force:
        try:
            rows = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
        except OSError:
            rows = -1
        if abs(rows - n_sec) <= 1:
            log("= %s: features.csv уже есть (%d строк при %d с) — пропуск (--force чтобы пересчитать)"
                % (name, rows, n_sec))
            return 0
        log("! %s: features.csv: %d строк при ожидаемых %d — пересчитываю" % (name, rows, n_sec))

    log("== %s: %.1f с аудио, %d секунд сетки, батч %d, устройство %s" % (name, dur, n_sec, batch, device))
    t_all = time.time()
    ensure_hf_cache()

    t0 = time.time()
    raw_db = rms_db_per_sec(audio, n_sec)
    voice_db = rms_db_per_sec(voice, n_sec)
    log("  RMS: raw_db/voice_db за %.1f с" % (time.time() - t0))

    x = read_int16(voice)
    with open(src_dir / "words.json", encoding="utf-8") as f:
        words = json.load(f)
    word_is_marker = np.array([norm_word(w["w"]) in set(MARKERS) for w in words], dtype=bool)
    word_t = np.array([w["start"] for w in words], dtype=np.float64)
    n_words = np.zeros(n_sec, dtype=np.int64)
    n_markers = np.zeros(n_sec, dtype=np.int64)
    if len(word_t):
        sec = word_t.astype(np.int64)
        keep = (sec >= 0) & (sec < n_sec)
        n_words += np.bincount(sec[keep], minlength=n_sec)[:n_sec]
        n_markers += np.bincount(sec[keep & word_is_marker], minlength=n_sec)[:n_sec]

    m = load_models(device)
    t0 = time.time()
    dims = np.zeros((n_sec, 3), dtype=np.float64)
    ast_v = np.zeros((n_sec, 2), dtype=np.float64)
    emo_v = np.zeros((n_sec, len(m["emo_cols"])), dtype=np.float64)
    next_mark, done = 0, 0
    for s in range(0, n_sec, batch):
        ts = list(range(s, min(s + batch, n_sec)))
        a, d, v, laugh, scream, e_probs = run_batch(m, windows(x, ts))
        e = ts[-1] + 1
        dims[s:e, 0], dims[s:e, 1], dims[s:e, 2] = a, d, v
        ast_v[s:e, 0], ast_v[s:e, 1] = laugh, scream
        emo_v[s:e] = e_probs
        done = e
        if done >= next_mark or done == n_sec:
            log("  признаки: %d%% (%d/%d с), %.0f с счёта" % (100 * done // n_sec, done, n_sec, time.time() - t0))
            next_mark = done + max(1, (n_sec * 5) // 100)

    data = {"t": np.arange(n_sec, dtype=np.int64),
            "raw_db": raw_db, "voice_db": voice_db,
            "arousal": dims[:, 0], "dominance": dims[:, 1], "valence": dims[:, 2]}
    for i, c in enumerate(m["emo_cols"]):
        data[c] = emo_v[:, i]
    data["laugh"] = ast_v[:, 0]
    data["scream"] = ast_v[:, 1]
    data["words"] = n_words
    data["markers"] = n_markers
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False, float_format="%.6f")
    log("[features] %s: %.1f с всего (модели %.1f с), %d строк x %d столбцов -> %s"
        % (name, time.time() - t_all, time.time() - t0, len(df), len(df.columns), csv_path.name))
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Посекундные признаки (3-с окно) из voice.wav")
    ap.add_argument("name", help="имя исходника без расширения")
    ap.add_argument("--force", action="store_true", help="пересчитать даже при готовом features.csv")
    ap.add_argument("--batch", type=int, default=32, help="размер батча (по умолчанию 32)")
    ap.add_argument("--device", default="cuda", help="cuda или cpu")
    ap.add_argument("--out-root", default=str(ANALYSIS),
                    help="куда класть <имя>/features.csv (по умолчанию analysis_dir)")
    args = ap.parse_args(argv)
    try:
        return compute(args.name, args.force, args.batch, args.device, Path(args.out_root))
    except Exception as ex:
        import traceback
        traceback.print_exc()
        log("! %s: %s" % (args.name, ex))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
