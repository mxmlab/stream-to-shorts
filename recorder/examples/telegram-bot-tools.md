# Example: recording section for a Telegram bot (agent tools)

A drop-in section for the tool description of an agent that runs on the same machine as
`streamrec` (for example an OpenClaw-style Telegram bot). It is **anonymised**: channel,
paths, streamer name and the personal recording folder are gone; only the interface remains.

The bot needs no Twitch API key and no OAuth token: it just runs four commands. The channel is
whatever `STREAMREC_CHANNEL` says in `~/streamrec/.env`, so this text can be shipped as is.

```markdown
## Stream recording (recorder/)
- CLI: `~/streamrec/streamrec` — run it as is, from any directory.
- "start recording" → `~/streamrec/streamrec start`. Starts recording the live channel in OBS;
  if OBS is off it brings it up itself (up to ~60 s). The channel is taken from
  `STREAMREC_CHANNEL` in `~/streamrec/.env`.
- "stop recording" → `~/streamrec/streamrec stop`. Stops the recording and prints the file path.
- "is it recording?" → `~/streamrec/streamrec status`. Whether it records, for how long, and
  whether the channel is live.
- Recording files: `$STREAMREC_OUTDIR` (default `~/Videos/streams`), Hybrid MP4, names like
  `2026-09-25 20-21-17.mp4`.
- Automation: the `streamrec-watch.timer` user timer runs every 60 s, starts recording when the
  channel goes live and stops it after the stream ends. Normally nothing has to be done by hand —
  the commands above are only for explicit requests.
- If `start` complains about obs-websocket, OBS did not come up — check `~/streamrec/streamrec.log`.
- Do not pass a channel name as an argument: the channel lives in `~/streamrec/.env`.
```

Notes for whoever wires this up:

- The timer records unattended; the bot commands are for status and manual intervention only.
- `start` marks the recording as `manual`, so the watcher will not stop it when the channel
  goes offline; use `stop` explicitly in that case.
- `status` prints the **previous** file while a recording is running — the current path is only
  known after `stop`.
- Keep the section short in the tool description: every extra sentence burns context on
  every bot turn.
