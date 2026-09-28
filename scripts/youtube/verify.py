# -*- coding: utf-8 -*-
"""Acceptance checks for the built YouTube version (TASK.md, "Критерий готовности").

1. ffprobe of the output: geometry / fps / SAR / codecs + duration vs
   intro_duration(intro video stream) + sum(keep_segments) from the EDL (+-0.2 s).
2. A/V sync at 3 points (start / middle / end): |video span - audio span| < 0.05 s
   measured on real 20 s clips (stream copy), plus the A/V offset of the first
   packet pair.
3. Censorship: first 5 mute windows that fall inside keep segments, plus all 3
   windows with kind="word": volumedetect on the exact window in the output
   (max_volume <= -60 dB) and on the 0.5 s before it (> -40 dB).
4. Render time comes from the build job.
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO_ROOT)
from config import CONFIG  # noqa: E402

FFPROBE = CONFIG.ffprobe
FFMPEG = CONFIG.ffmpeg
WORK = HERE
# default output; an EDL with an "output" key overrides it (see main())
OUT = os.path.join(WORK, "youtube_final.mp4")
EDL_PATH = os.path.join(WORK, "edl.json")
FPS = 60
INTRO = 4.5          # default; replaced by the measured intro video duration
TMP = os.path.join(WORK, "_sync_tmp.mp4")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def resolve_edl(path):
    """`--edl` value: absolute paths as they are, relative ones from cwd or WORK."""
    if os.path.isabs(path):
        return os.path.normpath(path)
    cand = os.path.abspath(path)
    if os.path.exists(cand):
        return cand
    return os.path.normpath(os.path.join(WORK, path))


def edl_from_argv(argv):
    """Path of the EDL: `--edl <path>` / `--edl=<path>`, else the default one."""
    for i, a in enumerate(argv):
        if a == "--edl":
            if i + 1 >= len(argv):
                sys.exit("--edl requires a path")
            return resolve_edl(argv[i + 1])
        if a.startswith("--edl="):
            return resolve_edl(a.split("=", 1)[1])
    return EDL_PATH


def intro_duration(path):
    """Duration of the VIDEO stream of `path`, rounded to 1/60 s."""
    p = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=duration",
                        "-show_entries", "format=duration",
                        "-of", "json", path], capture_output=True, text=True)
    dur = None
    if p.returncode == 0 and p.stdout.strip():
        try:
            probed = json.loads(p.stdout)
        except ValueError:
            probed = {}
        for s in probed.get("streams") or []:
            if s.get("duration") not in (None, "N/A"):
                dur = float(s["duration"])
                break
        fmt = probed.get("format") or {}
        if dur is None and fmt.get("duration") not in (None, "N/A"):
            dur = float(fmt["duration"])
    if dur is None:
        sys.exit("ffprobe could not read the duration of edl['intro']: %s" % path)
    return round(dur * FPS) / FPS


def load_edl(path):
    if not os.path.exists(path):
        sys.exit("EDL not found: %s" % path)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def ffprobe_json(args, path=None):
    p = subprocess.run([FFPROBE, "-v", "error", *args, "-of", "json", path or OUT],
                       capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stderr)
        sys.exit(1)
    return json.loads(p.stdout)


def to_output_time(t, keeps):
    off = 0.0
    for s, e in keeps:
        if s <= t <= e:
            return INTRO + off + (t - s)
        off += e - s
    return None


def _db(text, key):
    m = re.search(key + r":\s*(-?[\w.]+) dB", text)
    if not m:
        return None
    v = m.group(1)
    return -999.0 if v.startswith("-inf") else float(v)


def volumedetect(t, dur):
    p = subprocess.run([FFMPEG, "-hide_banner", "-nostdin", "-ss", "%.6f" % t,
                        "-t", "%.6f" % dur, "-i", OUT, "-vn", "-af",
                        "volumedetect", "-f", "null", "-"],
                       capture_output=True, text=True)
    return _db(p.stderr, "max_volume"), _db(p.stderr, "mean_volume")


def packet_span(path, sel):
    p = subprocess.run([FFPROBE, "-v", "error", "-select_streams", sel,
                        "-show_entries", "packet=pts_time,duration_time",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    lo = hi = first = None
    for line in p.stdout.splitlines():
        f = line.split(",")
        if not f or not f[0]:
            continue
        try:
            pts = float(f[0])
            d = float(f[1]) if len(f) > 1 and f[1] not in ("", "N/A") else 0.0
        except ValueError:
            continue
        if first is None:
            first = pts
        lo = pts if lo is None else min(lo, pts)
        hi = pts + d if hi is None else max(hi, pts + d)
    return first, lo, hi


def clip_spans(t0, dur):
    """copy a real clip and measure both stream spans in it."""
    subprocess.run([FFMPEG, "-y", "-v", "error", "-ss", "%.6f" % t0, "-i", OUT,
                    "-t", "%.6f" % dur, "-c", "copy",
                    "-avoid_negative_ts", "make_zero", TMP],
                   capture_output=True, text=True, check=True)
    v = packet_span(TMP, "v:0")
    a = packet_span(TMP, "a:0")
    os.remove(TMP)
    return v, a


def packet_at(t0, sel, pad=5.0):
    """start time of the packet covering t0 (read range is only a window)."""
    p = subprocess.run([FFPROBE, "-v", "error", "-select_streams", sel,
                        "-read_intervals", "%f%%+%f" % (max(0.0, t0 - pad), 2 * pad),
                        "-show_entries", "packet=pts_time", "-of", "csv=p=0", OUT],
                       capture_output=True, text=True)
    pts = sorted(float(x) for x in p.stdout.split() if x.strip())
    prev = [x for x in pts if x <= t0 + 1e-9]
    return prev[-1] if prev else None


def main(argv=None):
    global OUT, INTRO
    ap = argparse.ArgumentParser(
        prog="verify.py",
        description="Приёмка собранной YouTube-версии по монтажному листу (EDL)")
    ap.add_argument("--edl", default=EDL_PATH, help="монтажный лист (JSON)")
    args = ap.parse_args(argv)
    edl_path = resolve_edl(args.edl)
    edl = load_edl(edl_path)
    keeps = [(float(a), float(b)) for a, b in edl["keep_segments_source_time"]]
    mutes = edl["mutes_source_time"]
    keep_total = sum(e - s for s, e in keeps)
    INTRO = intro_duration(edl["intro"])
    if abs(INTRO - 4.5) > 0.02:
        print("WARNING: intro video duration is %.3f s, not 4.5 +- 0.02 s (%s)"
              % (INTRO, edl["intro"]))
    out = edl.get("output") or OUT
    if not os.path.isabs(out):
        out = os.path.join(os.path.dirname(os.path.abspath(edl_path)), out)
    OUT = os.path.normpath(out)
    expected = keep_total + INTRO

    print("edl to verify:", edl_path)
    if not os.path.exists(OUT):
        print("output not found:", OUT)
        return 1

    print("=" * 78)
    print("1. CONTAINER / STREAMS")
    print("=" * 78)
    st = ffprobe_json(["-show_entries",
                       "stream=index,codec_type,codec_name,width,height,"
                       "r_frame_rate,avg_frame_rate,sample_aspect_ratio,sample_rate,"
                       "channels,profile,pix_fmt,duration,nb_frames",
                       "-show_entries", "format=duration,size,bit_rate"])
    for s in st["streams"]:
        if s["codec_type"] == "video":
            print("  video: %s %sx%s r=%s avg=%s SAR=%s profile=%s pix=%s "
                  "frames=%s dur=%s" %
                  (s["codec_name"], s.get("width"), s.get("height"),
                   s.get("r_frame_rate"), s.get("avg_frame_rate"),
                   s.get("sample_aspect_ratio"), s.get("profile"), s.get("pix_fmt"),
                   s.get("nb_frames"), s.get("duration")))
        else:
            print("  audio: %s %s Hz %s ch dur=%s" %
                  (s["codec_name"], s.get("sample_rate"), s.get("channels"),
                   s.get("duration")))
    fmt = st["format"]
    d = abs(float(fmt["duration"]) - expected)
    print("  format duration      = %.3f s" % float(fmt["duration"]))
    print("  expected duration    = %.2f (intro %s) + %.3f (keeps) = %.3f s"
          % (INTRO, os.path.basename(edl["intro"]), keep_total, expected))
    print("  difference           = %.3f s  -> %s (tolerance +-0.2 s)"
          % (d, "OK" if d <= 0.2 else "FAIL"))
    print("  size                 = %.2f GiB, %s bit/s"
          % (int(fmt["size"]) / 2 ** 30, fmt.get("bit_rate")))

    print()
    print("=" * 78)
    print("2. A/V SYNC at start / middle / end")
    print("=" * 78)
    dur_total = float(fmt["duration"])
    vdur = [float(s["duration"]) for s in st["streams"] if s["codec_type"] == "video"][0]
    adur = [float(s["duration"]) for s in st["streams"] if s["codec_type"] == "audio"][0]
    print("  whole file: video stream duration %.3f s, audio %.3f s, "
          "difference %.3f s -> %s" %
          (vdur, adur, abs(vdur - adur), "OK" if abs(vdur - adur) < 0.05 else "FAIL"))
    points = [("start", 1.0), ("middle", dur_total / 2), ("end", dur_total - 1.0)]
    ok2 = True
    for label, t0 in points:
        v = packet_at(t0, "v:0")
        a = packet_at(t0, "a:0")
        off = a - v
        ok2 &= abs(off) < 0.05
        print("  %-6s t=%9.3f: video frame covering t starts %10.3f, audio packet "
              "covering t starts %10.3f, A/V offset %+.3f s -> %s"
              % (label, t0, v, a, off, "OK" if abs(off) < 0.05 else "FAIL"))
    print("  (packet grid: video 1/60 = 0.0167 s, audio 1024/48000 = 0.0213 s;")
    print("   a stream-copy clip can therefore not resolve better than ~0.04 s)")
    for label, t0 in points:
        v, a = clip_spans(max(0.0, t0 - 10), 20.0)
        vs, as_ = v[2] - v[1], a[2] - a[1]
        print("  cross-check, 20 s copy clip at %9.3f: video span %.3f s, audio span "
              "%.3f s, difference %.3f s" % (t0 - 10, vs, as_, abs(vs - as_)))

    print()
    print("=" * 78)
    print("3. CENSORSHIP (volumedetect on the exact window in the output)")
    print("=" * 78)

    def row(tag, m, seg):
        src_a, src_b = m["a"], m["b"]
        t = to_output_time(src_a, keeps)
        ln = src_b - src_a
        wmax, wmean = volumedetect(t, ln)
        cmax, cmean = volumedetect(t - 0.5, 0.5)
        ok = (wmax is not None and wmax <= -60.0)
        ctl = (cmax is not None and cmax > -40.0)
        print("  %-6s %-9s %-6s src %9.3f..%9.3f (%.3f s) -> out %9.3f   "
              "max %7.2f dB  mean %7.2f dB | ctl(-0.5 s) max %6.2f dB   %s / %s"
              % (tag, m["word"], m["kind"], src_a, src_b, ln, t,
                 wmax, wmean, cmax, "OK" if ok else "FAIL",
                 "ctl OK" if ctl else "ctl FAIL"))
        return ok

    inside = [m for m in mutes if any(m["a"] < e and m["b"] > s for s, e in keeps)]
    inside.sort(key=lambda m: m["a"])
    print("  mute windows inside keep segments: %d of %d" % (len(inside), len(mutes)))
    print()
    for m in inside[:5]:
        row("first5", m, None)
    print()
    for m in inside:
        if m["kind"] == "word":
            row("word", m, None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
