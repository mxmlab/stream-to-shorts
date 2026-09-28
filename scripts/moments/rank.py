# -*- coding: utf-8 -*-
"""Шаг 3: рейтинг моментов по посекундным признакам.

    python rank.py [--features-root <папка>] [--sources <имя> ...] [--out-dir <папка>]

Для каждого исходника: маска голоса (voiced = voice_db > -50) — на не-voiced секундах arousal, dominance,
valence, emo_*, laugh, scream обнуляются в NaN ДО робастного z, поэтому медиана/MAD считаются только по
voiced (модели эмоций врут на тишине: при voice_db=-200 arousal ~0.7, и пики уезжали в паузы).
Дальше робастный z ((x-median)/(1.4826*MAD), NaN->0, клип [-3,6]), сглаживание скользящим средним 5 с,
пики — локальные максимумы score с минимумом 45 с между пиками, top-N по score.

Пишет в `--out-dir` (по умолчанию <out_dir>/moments): candidates.md и candidates.json
(конфигурация G, top-20, с колонкой peak_voice_db — максимум voice_db в окне +-3 с).

Необязательная сверка с эталоном: если рядом лежат модули `cut_clips.py` / `cut_gas.py`
(списки CLIPS с вручную отобранными клипами), строится ещё и eval.md — таблица попаданий
пиков в эталонные клипы. Без этих модулей скрипт работает: пишет только candidates.*.
ffmpeg не запускается.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent                  # scripts/moments
REPO = BASE.parent.parent                               # корень репозитория
sys.path.insert(0, str(REPO))
from config import CONFIG                               # noqa: E402

ANALYSIS = Path(CONFIG.analysis_dir)

CONFIGS = {
    "A_raw": {"raw_db": 1.0},
    "B_voice": {"voice_db": 1.0},
    "C_voice_arousal": {"voice_db": 1.0, "arousal": 1.0},
    "D_laugh_scream": {"voice_db": 1.0, "laugh": 1.0, "scream": 1.0},
    "E_all": {"voice_db": 1.0, "arousal": 1.0, "laugh": 1.0, "scream": 1.0,
              "emo_positive": 0.5, "emo_angry": 0.5, "markers": 0.5},
    "F_all_novalence": {"voice_db": 1.0, "arousal": 1.0, "laugh": 1.0, "scream": 1.0, "markers": 0.5},
    "G_voice_arousal_laugh_scream_pos": {"voice_db": 1.0, "arousal": 1.0, "scream": 1.0, "laugh": 1.0,
                                         "markers": 0.5, "emo_positive": 0.5},
}
CAND_CFG = "G_voice_arousal_laugh_scream_pos"   # конфигурация для candidates.md/.json

VOICED_DB = -50.0       # порог маски голоса: voice_db > VOICED_DB — секунда считается voiced
MASK_COLS = ["arousal", "dominance", "valence", "emo_angry", "emo_sad", "emo_neutral", "emo_positive",
             "laugh", "scream"]        # эти признаки на не-voiced секундах -> NaN (до робастного z)

SMOOTH_SEC = 5          # скользящее среднее, с
MIN_GAP_SEC = 45.0      # минимум между пиками, с
PAD_SEC = 5.0           # допуск попадания в эталонный клип, с
PEAK_VOICE_WIN = 3.0    # окно +-3 с для peak_voice_db в candidates
KS = (10, 20)
TOP_N = 20              # сколько пиков в candidates


def log(msg: str) -> None:
    print(msg, flush=True)


def sec(tc: str) -> int:
    h, m, s = (int(x) for x in tc.split(":"))
    return h * 3600 + m * 60 + s


def hhmmss(t: float) -> str:
    s = int(round(t))
    return "%02d:%02d:%02d" % (s // 3600, (s % 3600) // 60, s % 60)


def find_sources(features_root: Path) -> list[str]:
    """Исходники с готовым features.csv — по именам папок."""
    if not features_root.is_dir():
        return []
    out = [p.name for p in sorted(features_root.iterdir())
           if (p / "features.csv").is_file()]
    return out


def load_clips() -> dict[str, list[dict]]:
    """Эталон из необязательных модулей cut_clips.py / cut_gas.py (списки CLIPS).

    Модулей нет — сверка пропускается (пустой словарь). Формат CLIPS:
    {ключ: [(num, "HH:MM:SS", "HH:MM:SS", название), ...]}.
    """
    clips: dict[str, list[dict]] = {}
    for mod_name, fname in (("cut_clips", "cut_clips.py"), ("cut_gas", "cut_gas.py")):
        path = ANALYSIS / fname
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location(mod_name, str(path))
        if spec is None or spec.loader is None:
            raise RuntimeError("не импортируется %s" % path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)          # main() под if __name__ == "__main__"
        for key, items in mod.CLIPS.items():
            base = key.split("/")[0]
            folder = key if "/" in key else ""
            for num, a, b, name in items:
                clips.setdefault(base, []).append(
                    {"folder": folder, "num": num, "name": name, "start": sec(a), "end": sec(b)})
    return clips


def zscore(x: np.ndarray) -> np.ndarray:
    """Робастный z: (x-median)/(1.4826*MAD), NaN -> 0, клип [-3, 6]."""
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x), x, np.nan)
    if np.isnan(x).all():
        return np.zeros_like(x)
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        z = np.zeros_like(x)
    else:
        z = (x - med) / scale
    z = np.where(np.isfinite(z), z, 0.0)
    return np.clip(z, -3.0, 6.0)


def smooth(x: np.ndarray, win: int = SMOOTH_SEC) -> np.ndarray:
    """Скользящее среднее win секунд (по центру, края — по доступным значениям)."""
    return pd.Series(x).rolling(win, center=True, min_periods=1).mean().to_numpy()


def peaks(score: np.ndarray, gap: float = MIN_GAP_SEC, top: int = TOP_N) -> list[int]:
    """Локальные максимумы score, минимум gap секунд между пиками, top-N по score (по убыванию)."""
    n = len(score)
    cand = [i for i in range(n)
            if (i == 0 or score[i] >= score[i - 1]) and (i == n - 1 or score[i] >= score[i + 1])
            and ((i > 0 and score[i] > score[i - 1]) or (i < n - 1 and score[i] > score[i + 1]))]
    if not cand:                      # плоский score — берём все точки как кандидаты
        cand = list(range(n))
    kept: list[int] = []
    for i in sorted(cand, key=lambda i: (-score[i], i)):
        if all(abs(i - j) >= gap for j in kept):
            kept.append(i)
            if len(kept) >= top:
                break
    return kept


def mask_silence(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Маска голоса: voiced = voice_db > VOICED_DB.

    На не-voiced секундах в MASK_COLS ставится NaN ДО робастного z: np.nanmedian в zscore
    считает медиану/MAD только по voiced, а после z NaN -> 0 (как и раньше).
    raw_db, voice_db, words, markers не маскируются.
    """
    out = df.copy()
    voiced = out["voice_db"].to_numpy(dtype=np.float64) > VOICED_DB
    for c in MASK_COLS:
        if c in out.columns:
            out.loc[~voiced, c] = np.nan
    return out, voiced


def peak_voice_db(df: pd.DataFrame, t: int, win: float = PEAK_VOICE_WIN) -> float:
    """Максимум voice_db в окне [t-win, t+win] (сырые дБ, без z)."""
    v = df.loc[np.abs(df["t"].to_numpy() - t) <= win, "voice_db"].to_numpy(dtype=np.float64)
    return float(np.max(v)) if v.size else float("nan")


def md_cell(text: str) -> str:
    """Текст в ячейку markdown-таблицы: без переводов строк и без сырых «|»."""
    one = " ".join(str(text).split()).replace("|", "\\|")
    return one if one else "—"


def load_features(name: str, root: Path = ANALYSIS) -> pd.DataFrame | None:
    p = root / name / "features.csv"
    if not p.is_file():
        log("! %s: нет features.csv (%s)" % (name, p))
        return None
    df = pd.read_csv(p)
    if not len(df):
        log("! %s: features.csv пустой" % name)
        return None
    return df


def words_text(words: list[dict], t: int, win: float = 10.0) -> str:
    return " ".join(w["w"] for w in words if t - win <= w["start"] <= t + win)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Рейтинг моментов по features.csv")
    ap.add_argument("--features-root", default=str(ANALYSIS),
                    help="где лежат <имя>/features.csv (по умолчанию analysis_dir)")
    ap.add_argument("--sources", nargs="*", default=None,
                    help="имена исходников (по умолчанию — все с готовым features.csv)")
    ap.add_argument("--out-dir", default=None,
                    help="куда писать candidates.md/json (по умолчанию <out_dir>/moments)")
    args = ap.parse_args(argv)
    features_root = Path(args.features_root)
    out_dir = Path(args.out_dir) if args.out_dir else (Path(CONFIG.out_dir) / "moments")
    out_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    sources = list(args.sources) if args.sources else find_sources(features_root)
    if not sources:
        log("! нет исходников с features.csv в %s (сначала features.py)" % features_root)
        return 2

    clips = load_clips()
    if clips:
        log("эталон: " + ", ".join("%s=%d" % (s, len(clips.get(s, []))) for s in sources)
            + " (всего %d)" % sum(len(clips.get(s, [])) for s in sources))
    else:
        log("эталонных CLIPS нет (cut_clips.py/cut_gas.py не найдены) — только candidates")

    feats: dict[str, pd.DataFrame] = {}
    smoothed: dict[str, dict[str, np.ndarray]] = {}
    scores: dict[str, dict[str, np.ndarray]] = {}
    peaks_by: dict[str, dict[str, list[int]]] = {}
    for name in sources:
        df = load_features(name, features_root)
        if df is None:
            continue
        masked, voiced = mask_silence(df)
        feats[name] = df
        log("  %s: voiced %d/%d (%.0f%%), не-voiced -> NaN в %d признаках"
            % (name, int(voiced.sum()), len(df), 100.0 * voiced.mean(), len(MASK_COLS)))
        z = {c: smooth(zscore(masked[c].to_numpy())) for c in df.columns if c != "t"}
        smoothed[name] = z
        scores[name] = {cfg: sum(w * z[c] for c, w in cols.items()) for cfg, cols in CONFIGS.items()}
        peaks_by[name] = {cfg: peaks(s) for cfg, s in scores[name].items()}

    # ------------------------------------------------------------------ eval.md
    if clips:
        def hit_names(t: float, name: str) -> list[str]:
            return [c["name"] for c in clips.get(name, []) if c["start"] - PAD_SEC <= t <= c["end"] + PAD_SEC]

        lines = ["# Сверка пиков с эталонными клипами",
                 "",
                 "Эталон — `CLIPS` из `cut_clips.py` и `cut_gas.py` в папке анализа "
                 "(серии и youtube-подпапки считаются эталоном своего исходника).",
                 "",
                 "Попадание: пик внутри `[начало−5 с, конец+5 с]` эталонного клипа. "
                 "Пики: локальные максимумы сглаженного (5 с) score, минимум 45 с между пиками, top-K по score.",
                 "Маска голоса: `voiced = voice_db > −50`; на не-voiced секундах `arousal`, `dominance`, `valence`, "
                 "`emo_*`, `laugh`, `scream` = NaN ДО робастного z (медиана/MAD — только по voiced), после z NaN → 0.",
                 "В ячейках — **попало/всего** эталонных клипов исходника; в строке ИТОГО — сумма попаданий / сумма эталонов.",
                 "",
                 "| исходник | эталонов | " + " | ".join("%s R@%d" % (c, k) for c in CONFIGS for k in KS) + " |",
                 "|---|---|" + "---|" * (len(CONFIGS) * len(KS))]

        totals = {cfg: {k: 0 for k in KS} for cfg in CONFIGS}
        total_refs = 0
        for name in sources:
            refs = clips.get(name, [])
            total_refs += len(refs)
            cells = []
            for cfg in CONFIGS:
                for k in KS:
                    top = peaks_by.get(name, {}).get(cfg, [])[:k]
                    got = sum(1 for c in refs
                              if any(c["start"] - PAD_SEC <= t <= c["end"] + PAD_SEC for t in top))
                    totals[cfg][k] += got
                    cells.append("%d/%d" % (got, len(refs)))
            lines.append("| %s | %d | %s |" % (name, len(refs), " | ".join(cells)))
        total_cells = []
        for cfg in CONFIGS:
            for k in KS:
                rec = totals[cfg][k] / total_refs if total_refs else 0.0
                total_cells.append("%d/%d (%.2f)" % (totals[cfg][k], total_refs, rec))
        lines.append("| **ИТОГО** | %d | %s |" % (total_refs, " | ".join(total_cells)))
        lines += ["",
                  "Конфигурации score (веса при z-признаках):",
                  ""] + ["- `%s`: %s" % (cfg, " + ".join(
                      ("%.2g·" % w if w != 1.0 else "") + c for c, w in cols.items()))
                      for cfg, cols in CONFIGS.items()] + [""]

        (out_dir / "eval.md").write_text("\n".join(lines), encoding="utf-8")
        log("eval.md готов")

    # ------------------------------------------------- candidates.md / .json
    cfg_c = CAND_CFG
    cols_c = CONFIGS[cfg_c]
    md = ["# Кандидаты в моменты — конфигурация `%s`" % cfg_c,
          "",
          "`%s`: " % cfg_c + " + ".join(("%.2g·" % w if w != 1.0 else "") + c for c, w in cols_c.items()) + ".",
          "Маска голоса: `voiced = voice_db > −50`; на не-voiced секундах `arousal`, `dominance`, `valence`, "
          "`emo_*`, `laugh`, `scream` = NaN ДО робастного z (медиана/MAD — только по voiced), после z NaN → 0.",
          "Пики: локальные максимумы сглаженного (5 с) score, минимум 45 с между пиками, top-20 по score.",
          "Вклады — вес × z (после сглаживания), пять самых больших."
          + (" «НОВЫЙ» — пик не попал ни в один эталонный клип." if clips else ""),
          "`peak_voice_db` — максимум `voice_db` в окне [t−3, t+3] (сырые дБ).",
          "Текст — слова из `words.json` с началом в окне [t−10, t+10].",
          ""]
    out_json: list[dict] = []
    for name in sources:
        if name not in feats:
            continue
        df = feats[name]
        words_path = features_root / name / "words.json"
        words = json.load(open(words_path, encoding="utf-8")) if words_path.is_file() else []
        md += ["## %s (эталонных клипов: %d)" % (name, len(clips.get(name, []))), "",
               "| # | время | score | peak_voice_db | вклады (вес × z) | эталон | текст words.json [t−10, t+10] |",
               "|---|---|---|---|---|---|---|"]
        for rank, i in enumerate(peaks_by[name][cfg_c], 1):
            t = int(df["t"].iloc[i])
            score = float(scores[name][cfg_c][i])
            contrib_all = {c: float(w * smoothed[name][c][i]) for c, w in cols_c.items()}
            contrib = dict(sorted(contrib_all.items(), key=lambda kv: kv[1], reverse=True)[:5])
            pvdb = peak_voice_db(df, t)
            hit = [c["name"] for c in clips.get(name, [])
                   if c["start"] - PAD_SEC <= t <= c["end"] + PAD_SEC]
            text = words_text(words, t)
            out_json.append({"src": name, "t": t, "score": round(score, 4),
                             "peak_voice_db": round(pvdb, 2),
                             "contrib": {c: round(v, 4) for c, v in contrib.items()},
                             "ref": hit[0] if hit else None, "text": text})
            md.append("| %d | `%s` | %.2f | %.2f | %s | %s | %s |" % (
                rank, hhmmss(t), score, pvdb,
                ", ".join("%s=%.2f" % (c, v) for c, v in contrib.items()),
                ("**" + hit[0] + "**" if hit else ("НОВЫЙ" if clips else "—")),
                md_cell(text)))
        md.append("")
    (out_dir / "candidates.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "candidates.json").write_text(
        json.dumps(out_json, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    log("candidates.md/json готовы: %d пиков -> %s" % (len(out_json), out_dir))
    log("rank.py: %.1f с" % (time.time() - t_all))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
