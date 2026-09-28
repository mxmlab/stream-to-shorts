# stream-to-shorts

A skill/pipeline that turns a recorded live stream into a YouTube version and a set of vertical
clips for TikTok / Shorts. It transcribes the stream (GigaAM for timing, Whisper for text),
detects profanity and mutes it, ranks the emotionally interesting moments, and renders 1080×1920
60 fps clips with burned-in captions. It is published as a skill for Claude Code and other
agents: `SKILL.md` carries the YAML frontmatter (`name`, `description`) and the working order.

Licensed under **AGPL-3.0**.

## Credits / Acknowledgements

Parts of the censoring and transcription code are adapted from
**Reelsi** (https://github.com/mxmlab/reelsi), AGPL-3.0.

## What it does

Two outputs from one recording:

1. **YouTube version** — the full stream with profanity censored by muting the audio windows of
   the bad words (video untouched), plus a short procedurally drawn intro.
2. **Vertical clips** — several 1080×1920, 60 fps clips with burned-in captions, punch-in zooms
   and normalized loudness, ready for TikTok / YouTube Shorts.

## Step 0: auto-recording (recorder/)

`recorder/` is the step before this pipeline: a small CLI plus a systemd user timer that watches
a Twitch channel and records the broadcast with OBS Studio over obs-websocket (Hybrid MP4,
NVENC, 1920×1080 60 fps). It starts recording when the channel goes live, stops it after the
stream ends, and never touches a recording started manually.

The only thing you configure is `~/streamrec/.env` (`STREAMREC_CHANNEL`, obs-websocket password,
`STREAMREC_OUTDIR`); no channel is hard-coded. Install with `recorder/install.sh`, point
`stream_root` at `STREAMREC_OUTDIR`, and the recordings become sources for
`scripts/sources.py`.

See **[recorder/README.md](recorder/README.md)** for requirements, step-by-step install,
commands and troubleshooting.

## Requirements

- **Python 3.10**
- **ffmpeg** built with NVENC (`h264_nvenc`) and CUDA support, plus `ffprobe`
- An **NVIDIA GPU** (transcription, source separation, feature extraction and rendering all use it)
- Python packages:
  - `gigaam` (model `v3_ctc`) — transcription and word-level timing
  - `faster-whisper` (model `large-v3`) — text and whole-stream profanity pass
  - `audio-separator` + the `vocals_mel_band_roformer.ckpt` model — vocal isolation
  - `numpy`, `pandas`, `soundfile`, `Pillow`
- Optional: **node** + `npx hyperframes` for the `hf` caption layer (`scripts/clips/hf_overlay.py`)
- Profanity dictionaries (`badwords.txt`, `okwords.txt`) — **not shipped**, see
  [Profanity dictionaries](#profanity-dictionaries)

## Install

```bash
git clone https://github.com/<owner>/stream-to-shorts.git
cd stream-to-shorts

pip install gigaam faster-whisper audio-separator numpy pandas soundfile Pillow

cp config.example.json config.json
# then edit config.json and fill in your paths
```

`config.json` is git-ignored. It is loaded by `config.py` from the repository root, unless the
environment variable **`STS_CONFIG`** points at another file:

```bash
export STS_CONFIG=/path/to/my-config.json     # Windows: $env:STS_CONFIG = "C:/cfg/sts.json"
```

If `config.json` is missing, you get a clear error instead of a crash in the middle of a render:

```
нет файла настроек: <repo>/config.json
скопируйте config.example.json в config.json и поправьте пути
(или укажите путь к своему конфигу в переменной окружения STS_CONFIG)
```

A missing required key gives a similar message naming the key and the config file.

## Configuration reference

Every key of `config.example.json`:

| Key | Meaning |
|---|---|
| `stream_root` | **Required.** Root folder with the source recordings. `scripts/sources.py` finds a recording by name here (in the root or in the game subfolder). |
| `analysis_dir` | Where per-source analysis results go; default `<stream_root>/_analysis`. |
| `clips_root` | Ready-made cuts for `scripts/clips/render_vertical.py`; default `<stream_root>/clips`. |
| `clips_vertical_root` | Output root for vertical renders; default `<stream_root>/clips_vertical`. |
| `out_dir` | Output folder; relative paths are resolved from the repo root. Default `out`. |
| `temp_dir` | Temporary files; default `_cache/temp`. |
| `badwords_path` | **Required.** Path to the bad-words list. |
| `okwords_path` | **Required.** Path to the false-positive safety list. |
| `ffmpeg` | **Required.** Path to the `ffmpeg` executable. |
| `ffprobe` | **Required.** Path to the `ffprobe` executable. |
| `separator_exe` | **Required.** `audio-separator` executable. |
| `separator_model_dir` | **Required.** Folder with the separation model. |
| `separator_model` | Model file name; default `vocals_mel_band_roformer.ckpt`. |
| `fonts_dir` | **Required.** Folder with fonts. |
| `font_caption` | **Required.** Font for clip captions. |
| `font_title` | **Required.** Font for the clip title plate. |
| `font_intro` | **Required.** Font for the intro disclaimer. |
| `font_intro_body` | Font for the intro body text; defaults to `font_intro`. |
| `censor.*` | Profanity-window rule, see below. |
| `layouts.*` | Webcam/game crop layouts, see below. |

### `layouts`

A mapping `layout_name -> {"cam": [w, h, x, y], "game_x0": int, "note": str}`:

- `cam` — the webcam rectangle **inside the 1920×1080 source frame** (`[width, height, x, y]`).
- `game_x0` — the left edge of the game crop; **the game crop is 960×1080**.
- `note` — free-form human comment.

Two example entries are provided out of the box: `example_cam_bottom_left` and
`example_cam_top_left`. An EDL picks a layout with its `"layout"` field; if the field is absent it
falls back to `"src"`. The layout may change mid-stream, so give every layout its own entry and set
the right one per EDL.

### `censor`

| Key | Default | Meaning |
|---|---|---|
| `window_frac` | `0.40` | Fraction of the bad word's duration used as the muted window. |
| `min_window_sec` | `0.15` | Lower bound for that window, in seconds. |
| `word_pad_sec` | `0.03` | Padding added on each side for the worst (whole-word) matches. |

Exactly what they mean:

- For a normal bad word the muted window is **centred on the middle letter of the word**, its
  length is `max(window_frac * word_duration, min_window_sec)`, and it is **clipped to the word
  bounds**.
- The **worst word** — one matched as a whole word — is muted **entirely**, plus `word_pad_sec`
  on each side.

## Quick start

1. Install and configure as above (`config.json` with `stream_root`, dictionaries, ffmpeg paths).
2. Put the recording under `stream_root` (root or a game subfolder) so `find_source(name)` sees it.
3. Transcribe and analyze it (GPU, roughly ~15 min for a 2.5 h stream):

   ```bash
   python scripts/analyze.py path/to/stream.mp4
   ```

   → `<analysis_dir>/<name>/`: `audio.wav`, `words.json`, `transcript.txt`, `loudness.json`,
   `peaks.txt`.

4. Isolate the voice and compute per-second features:

   ```bash
   python scripts/moments/separate.py <name>
   python scripts/moments/features.py <name>
   ```

5. Rank candidate moments:

   ```bash
   python scripts/moments/rank.py
   ```

6. Build the mute windows for the YouTube version:

   ```bash
   python scripts/yt_mutes.py <src> --rule tt --edl examples/edl_youtube.json --report
   ```

7. Build and verify the full YouTube version:

   ```bash
   python scripts/youtube/build.py --edl examples/edl_youtube.json --dry-run
   python scripts/youtube/build.py --edl examples/edl_youtube.json
   python scripts/youtube/verify.py --edl examples/edl_youtube.json
   ```

8. Write the clip EDLs (one per selected moment; see `examples/edl_clip.json`), then fix the
   subtitle text/timing:

   ```bash
   python scripts/clips/fix_words.py examples/edl_clip.json
   ```

9. Tighten the cuts and render the vertical clip:

   ```bash
   python scripts/clips/tighten.py --in <edl-dir> --out <edl-dir>-tight
   python scripts/clips/render_tt.py <edl-dir>-tight/<name>.json --sheet
   ```

   Inspect the contact sheet: captions must not sit on the game's own text, the webcam layout must
   be right, loudness must land at −14 ±1 LUFS.

## Profanity dictionaries

**The dictionaries are NOT shipped with this repository** — `badwords.txt` and `okwords.txt` are in
`.gitignore`. You must supply your own.

Ready-made Russian lists:
`https://github.com/mxmlab/reelsi/blob/main/data/badwords.txt` and
`https://github.com/mxmlab/reelsi/blob/main/data/okwords.txt`.
Point `badwords_path` / `okwords_path` in `config.json` at them (or at your own copies).

### Format, exactly as the code implements it

```python
def _parse_stems(lines):
    out = []
    for ln in lines:
        s = ln.split("#", 1)[0].strip().lower().replace("ё", "е")
        if s:
            out.append(s)
    return out

def _norm_word(s):
    return re.sub(r"[^\w]+", "", (s or "").lower().replace("ё", "е"), flags=re.UNICODE)

def is_bad(word, bad, ok):
    low = _norm_word(word)
    if not low:
        return False
    if any(o in low for o in ok):
        return False
    return any(b in low for b in bad)
```

So, precisely:

- **One stem per line.** Everything after `#` on a line is a comment and is stripped.
- Lines are lowercased and `ё` is folded to `е`; empty lines are skipped.
- A candidate word is normalized the same way: lowercased, `ё` → `е`, and **everything that is not
  `\w` removed** (punctuation, spaces, dashes).
- Matching is by **SUBSTRING**: the word is bad if it contains any bad stem as a substring,
  **UNLESS** it also contains any okword stem as a substring. **Okwords win over badwords and are
  checked first.**

Example: with `badwords.txt` containing `бля` and `okwords.txt` containing `бляха`:

- `бляха` is **clean** (okword wins);
- `блядь` is **muted** and shown with a `*` instead of the middle letter.

**Sections in `badwords.txt`:** the loader takes **only the lines AFTER the last line that starts
with `# --- мат`**. If no such line is present, the whole file is used. This lets one file carry
several themed sections while only the profanity one is applied.

### Prompt to generate a list for your language

Copy-paste into any LLM:

```
Generate two plain-text profanity dictionaries for the language: <LANGUAGE>.

Output format (strict):
- one stem per line, lowercase, no numbering, no bullet points, no explanations, no blank
  sections — just the lines;
- `#` starts a comment; use comment lines to label sections;
- the two lists must be clearly separated: first the badwords list, then a line
  `# --- okwords` followed by the okwords list.

Rules:
- Entries are matched as SUBSTRINGS of a normalized word (lowercase, everything that is
  not a letter/digit removed). Therefore prefer SHORT ROOTS over full words
  (e.g. a 3-4 letter root), so inflected forms are covered.
- badwords: roots of profanity, slurs and strong obscenities.
- okwords: a separate list of FALSE-POSITIVE safety words — ordinary words that merely
  CONTAIN a bad root as a substring. Okwords are checked first and always win.

Return only the two lists.
```

After generating: run the lists over your own transcripts, see what actually gets muted, and move
the false positives into `okwords.txt`. Do not edit the lists of other people silently — okword
changes are a user decision.

## EDL examples

Templates live in `examples/` (see `examples/edl_youtube.json` and `examples/edl_clip.json`).
An EDL is a JSON document.

**YouTube EDL** (what to cut, censor windows, intro) — the real key names used by
`scripts/youtube/build.py` and `scripts/yt_mutes.py`:

```json
{
  "name": "example_youtube",
  "source": "D:/streams/example_game/example_game1.mp4",
  "source_duration": 7200.0,
  "intro": "D:/streams/example_game/youtube/disclaimer.mp4",
  "intro_gain_db": -6.0,
  "output": "D:/streams/example_game/youtube/example_game1_final.mp4",
  "cuts_note": [{"a": 0.0, "b": 120.0, "why": "OBS setup on screen"}],
  "mutes_source_time": [{"a": 605.12, "b": 605.27, "word": "example", "kind": "letter"}],
  "keep_segments_source_time": [[120.0, 3300.0], [3340.0, 7188.0]]
}
```

`source`, `intro` and `output` are paths; `keep_segments_source_time` is a list of `[a, b]` pairs
in source seconds, `mutes_source_time` the censored windows (`scripts/yt_mutes.py` fills them in),
`intro_gain_db` the intro volume. `cuts_note` is a human-readable record of what was cut and why.

**Clip EDL** (one vertical clip) — the keys `scripts/clips/render_tt.py` reads:

```json
{
  "name": "example_clip",
  "src": "example_game1",
  "title": "Example title",
  "layout": "example_cam_bottom_left",
  "segments": [{"a": 1000.2, "b": 1003.4}, {"a": 1004.1, "b": 1008.9}],
  "punch": [{"t": 1013.1, "dur": 0.7, "scale": 1.15, "focus": "cam", "boom": true}],
  "cam_alt": true,
  "caption_y": 1300,
  "caption_off": [],
  "extra_mute": [],
  "game_view": {"crop": [480, 270, 960, 540]}
}
```

Field meanings: `src` is the source recording name (`scripts/sources.py`), `segments` are the kept
ranges in source seconds, `punch` are zoom-in moments (`focus` is `cam` or `game`, `boom` adds the
impact sound), `title` is the top plate, `caption_y` the caption baseline (move it away from the
game's own text), `layout` selects an entry from `layouts` in the config (falls back to `"src"`),
`cam_alt` alternates the webcam zoom on odd segments, `caption_off` suppresses captions in given
ranges, `extra_mute` adds manual mute windows, `game_view` crops the game out of an arbitrary
source rectangle, and `game_broll` (`{"t": <source second>}`) substitutes a moving shot from the
same layout when the on-screen game is static but the person is talking.

Rendering commands: `scripts/clips/render_tt.py` (main engine, EDL → vertical clip) and
`scripts/clips/render_vertical.py` (simpler renderer over pre-cut clips).

## По-русски

**stream-to-shorts** — навык и конвейер, который превращает запись стрима в две вещи:
полную **YouTube-версию** с зацензуренным матом (звук глушится в окнах плохих слов) и короткие
**вертикальные клипы 1080×1920, 60 fps** с вжёнными субтитрами для TikTok / Shorts.

Что нужно: Python 3.10, ffmpeg с NVENC и CUDA, видеокарта NVIDIA, пакеты `gigaam`,
`faster-whisper`, `audio-separator`, `numpy`, `pandas`, `soundfile`, `Pillow`. Словари мата в
репозиторий не входят — свои `badwords.txt` и `okwords.txt` укажите в `config.json`
(готовые русские списки есть в Reelsi).

Быстрый старт: `pip install ...` → скопировать `config.example.json` в `config.json` и прописать
пути → `python scripts/analyze.py <video>` → `separate.py` → `features.py` → `rank.py` →
`yt_mutes.py` → `youtube/build.py` + `verify.py` → EDL клипов → `clips/fix_words.py` →
`clips/tighten.py` → `clips/render_tt.py`. Подробный порядок — в `SKILL.md`.

Лицензия — **AGPL-3.0**. Часть кода цензуры и транскрипции адаптирована из
Reelsi (https://github.com/mxmlab/reelsi), AGPL-3.0.

**Шаг 0 — автозапись.** Саму запись стрима делает `recorder/`: CLI + пользовательский таймер
systemd, которые через obs-websocket поднимают OBS, начинают запись, когда канал выходит
в эфир, и останавливают её после конца эфира. Канал задаётся только в `~/streamrec/.env`
(`STREAMREC_CHANNEL`), папка записей — `STREAMREC_OUTDIR` (по умолчанию `~/Videos/streams`,
её же надо указать в `stream_root`). Установка — `recorder/install.sh`, подробности —
в [recorder/README.md](recorder/README.md).

## License

**AGPL-3.0** — see [LICENSE](LICENSE).

Adapted from Reelsi (https://github.com/mxmlab/reelsi), AGPL-3.0.
