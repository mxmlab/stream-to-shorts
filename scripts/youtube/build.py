# -*- coding: utf-8 -*-
"""YouTube version of a stream, built from an EDL.

    python build.py --edl <edl.json> [--dry-run] [--test] [--smoke]

One single ffmpeg invocation with a filter_complex graph, NVENC encode,
progress written to build.log (next to the EDL).

Pipeline:
  1. censorship on SOURCE timecode: sound of the source inside every
     mutes_source_time window [a, b] is set to volume=0 (-100 dB);
  2. keep_segments_source_time are cut out and glued back to back
     (video and audio cut identically, frame/sample exact);
  3. intro (volume = intro_gain_db) is placed BEFORE the assembled stream;
  4. encode h264_nvenc p5 vbr cq19 / AAC 192k 48 kHz stereo, +faststart.

Implementation note on step 1: ffmpeg evaluates the "enable" timeline option
once per audio frame (1024 samples = 21.3 ms for AAC), so a single
volume=0:enable='between(t,a,b)' over the whole source leaves up to 21.3 ms of
the censored word audible at the leading edge of each window (measured: -35 dB
instead of <= -60 dB).  Therefore the source audio is cut at the exact window
boundaries and volume=0:enable='between(t,0,<window length>)' is applied to the
exact [a, b] piece - the window itself is neither shifted nor extended.
(verified: the silent run in the assembled audio is exactly the requested
window, sample exact.)

Filter graph rule: every filter output pad feeds exactly ONE consumer (a pad
reused by several filters makes ffmpeg link only one of them, which silently
truncated the audio track in the first attempt).

ffmpeg/ffprobe come from config.json; the output file is `edl["output"]`
(an absolute path or a path relative to the EDL folder); without that key a
default next to the script is used.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO_ROOT)
from config import CONFIG  # noqa: E402

try:                                   # консоль Windows по умолчанию не utf-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

FFMPEG = CONFIG.ffmpeg
FFPROBE = CONFIG.ffprobe
WORK = HERE
# default EDL; `--edl <path>` overrides it.  build.log always sits next to the EDL.
EDL_PATH = os.path.join(WORK, "edl.json")
FILTER_SCRIPT = os.path.join(WORK, "build_filter.txt")

OUT = os.path.join(WORK, "youtube_final.mp4")

FPS = 60
SR = 48000
# the intro duration is measured with ffprobe (rounded to 1/60 s); this is only
# the value the measurement is expected (and warned) against
INTRO_DUR = 4.5

# reduced EDL for --test: one keep segment with three mute windows inside it
TEST_KEEP = [[1793.0, 2000.0]]


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


def build_graph(edl, intro_dur):
    """Return (filter_graph, stats)."""
    intro_gain = float(edl["intro_gain_db"])
    keeps = [(float(a), float(b)) for a, b in edl["keep_segments_source_time"]]
    mutes = [(float(m["a"]), float(m["b"])) for m in edl["mutes_source_time"]]

    # --- source audio pieces: (kind, start, end) in source seconds
    pieces = []
    for s, e in keeps:
        cur = s
        inside = sorted((max(a, s), min(b, e))
                        for a, b in mutes if a < e and b > s)
        for a, b in inside:
            if a > cur:
                pieces.append(("open", cur, a))
            pieces.append(("mute", a, b))
            cur = b
        if cur < e:
            pieces.append(("open", cur, e))

    parts = []
    # intro (video + audio), normalised to the same tb / SAR / format
    parts.append("[0:v]trim=end=%.3f,setpts=PTS-STARTPTS,setsar=1,settb=1/60[iv]"
                 % intro_dur)
    parts.append("[0:a]aformat=sample_fmts=fltp:sample_rates=48000:"
                 "channel_layouts=stereo,volume=%sdB,atrim=end_sample=%d,"
                 "asetpts=PTS-STARTPTS[ia]"
                 % (intro_gain, int(round(intro_dur * SR))))

    # video: one split output per keep segment, each consumed once
    nv = len(keeps)
    vlabels = ["[iv]"]
    parts.append("[1:v]split=%d%s" % (nv, "".join("[s%d]" % i for i in range(nv))))
    for i, (s, e) in enumerate(keeps):
        parts.append("[s%d]trim=start_frame=%d:end_frame=%d,setpts=PTS-STARTPTS,"
                     "setsar=1[v%d]" % (i, int(round(s * FPS)), int(round(e * FPS)), i))
        vlabels.append("[v%d]" % i)
    parts.append("%sconcat=n=%d:v=1:a=0[outv]" % ("".join(vlabels), nv + 1))

    # audio: one split output per piece, each consumed once
    alabels = ["[ia]"]
    if pieces:
        parts.append("[1:a]aformat=sample_fmts=fltp:sample_rates=48000:"
                     "channel_layouts=stereo,asplit=%d%s"
                     % (len(pieces), "".join("[g%d]" % i for i in range(len(pieces)))))
    for i, (kind, a, b) in enumerate(pieces):
        chain = ("atrim=start_sample=%d:end_sample=%d,asetpts=PTS-STARTPTS"
                 % (int(round(a * SR)), int(round(b * SR))))
        if kind == "mute":
            chain += ",volume=0:enable='between(t,0,%.9f)'" % (b - a)
        parts.append("[g%d]%s[p%d]" % (i, chain, i))
        alabels.append("[p%d]" % i)
    parts.append("%sconcat=n=%d:v=0:a=1[outa]" % ("".join(alabels), len(alabels)))

    graph = ";".join(parts)
    keep_total = sum(e - s for s, e in keeps)
    stats = {
        "keeps": keeps,
        "pieces": len(pieces),
        "mute_pieces": sum(1 for k, _, _ in pieces if k == "mute"),
        "keep_total": keep_total,
        "expected_duration": keep_total + intro_dur,
        "audio_segments": len(alabels),
        "graph_len": len(graph),
    }
    return graph, stats


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="build.py",
        description="YouTube-версия записи стрима из монтажного листа (EDL)")
    ap.add_argument("--edl", default=EDL_PATH, help="монтажный лист (JSON)")
    ap.add_argument("--dry-run", action="store_true", help="собрать граф, но не кодировать")
    ap.add_argument("--test", action="store_true", help="сокращённый EDL (один keep-сегмент)")
    ap.add_argument("--smoke", action="store_true", help="20 с в -f null (проверка графа)")
    args = ap.parse_args(argv)

    dry, smoke, test = args.dry_run, args.smoke, args.test
    edl_path = resolve_edl(args.edl)
    if not os.path.exists(edl_path):
        sys.exit("EDL not found: %s" % edl_path)
    log = os.path.join(os.path.dirname(os.path.abspath(edl_path)), "build.log")

    with open(edl_path, encoding="utf-8") as fh:
        edl = json.load(fh)
    if test:
        edl["keep_segments_source_time"] = TEST_KEEP
        edl["mutes_source_time"] = [
            m for m in edl["mutes_source_time"]
            if TEST_KEEP[0][0] <= m["a"] < TEST_KEEP[0][1]]

    intro_dur = intro_duration(edl["intro"])
    if abs(intro_dur - INTRO_DUR) > 0.02:
        print("WARNING: intro video duration is %.3f s, not %.1f +- 0.02 s (%s)"
              % (intro_dur, INTRO_DUR, edl["intro"]))

    out = edl.get("output") or OUT
    if not os.path.isabs(out):
        out = os.path.join(os.path.dirname(os.path.abspath(edl_path)), out)
    out = os.path.normpath(out)
    if test:
        out = os.path.splitext(out)[0] + "_test.mp4"
    graph, st = build_graph(edl, intro_dur)

    print("mode                    : %s" % ("TEST (reduced EDL)" if test else "FULL"))
    print("keep segments           : %d (%s)" % (len(st["keeps"]), st["keeps"]))
    print("sum of keep segments    : %.3f s" % st["keep_total"])
    print("intro                   : %.1f s, gain %.1f dB"
          % (intro_dur, edl["intro_gain_db"]))
    print("EXPECTED OUTPUT DURATION: %.3f s" % st["expected_duration"])
    print("mute pieces             : %d of %d audio pieces"
          % (st["mute_pieces"], st["pieces"]))
    print("filter graph length     : %d chars" % st["graph_len"])

    if st["graph_len"] > 30000:
        with open(FILTER_SCRIPT, "w", encoding="utf-8") as fh:
            fh.write(graph)
        graph_args = ["-filter_complex_script", FILTER_SCRIPT]
    else:
        graph_args = ["-filter_complex", graph]

    if dry:
        print("[dry-run] no encode")
        return 0

    cmd = [FFMPEG, "-y", "-hide_banner", "-nostdin", "-stats_period", "5",
           "-i", edl["intro"], "-i", edl["source"],
           *graph_args,
           "-map", "[outv]", "-map", "[outa]",
           "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19",
           "-b:v", "0", "-maxrate", "20M", "-bufsize", "40M",
           "-profile:v", "high", "-pix_fmt", "yuv420p", "-r", "60",
           "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
           "-movflags", "+faststart",
           "-progress", log, "-nostats"]
    if smoke:
        cmd += ["-t", "20", "-f", "null", "-"]
    else:
        cmd += [out]
    print("output                  : %s" % ("-f null (smoke)" if smoke else out))
    print("edl                     : %s" % edl_path)
    print("log                     : %s" % log)
    print("ffmpeg command length   : %d chars" % sum(len(c) + 1 for c in cmd))
    print("-" * 70, flush=True)
    rc = subprocess.call(cmd)
    print("-" * 70)
    print("ffmpeg exit code        : %d" % rc)
    if rc == 0 and os.path.exists(out):
        print("output size             : %d bytes" % os.path.getsize(out))
    return rc


if __name__ == "__main__":
    sys.exit(main())
