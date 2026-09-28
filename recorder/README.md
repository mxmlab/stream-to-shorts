# streamrec (auto-recording of Twitch streams via OBS)

`recorder/` is **Step 0** of the pipeline: it produces the recording that
`stream-to-shorts` later turns into a YouTube version and vertical clips.

It is a small CLI plus systemd timer that watches a Twitch channel and records the
broadcast with OBS Studio over obs-websocket:

- the timer runs once a minute; when the channel goes live, recording starts by itself;
- after 3 offline checks in a row the recording stops by itself;
- a recording started **manually** is never stopped by the watcher;
- a channel that is **empty** in the config produces a clear error — no channel is
  hard-coded anywhere in the script or in the OBS scene.

Nothing here is coupled to `stream-to-shorts`: the tool records, the pipeline processes.
Point `stream_root` in `config.json` at `STREAMREC_OUTDIR` and the recordings show up as
sources for `scripts/sources.py`.

## Requirements

- Ubuntu (tested on GNOME/Wayland; X11 also works, `QT_QPA_PLATFORM` is set to Wayland)
- NVIDIA GPU — the OBS profile encodes with `obs_nvenc_h264_tex`
- **OBS Studio as a Flatpak** — app id `com.obsproject.Studio`
  (`flatpak install flathub com.obsproject.Studio`).
  The apt build has no browser source, and the scene needs two of them.
- `python3` (3.10+) with `venv`, `systemd` with a user session
- Network access to Twitch GQL (`https://gql.twitch.tv/gql`) to check whether the channel is live
- Flatpak OBS has `ffprobe` **inside** the sandbox only:
  `flatpak run --command=ffprobe com.obsproject.Studio <file>` (no system-wide ffprobe needed)

## Install

```bash
git clone <this-repo> stream-to-shorts
cd stream-to-shorts/recorder
./install.sh              # add --force to overwrite an existing .env / OBS profile / OBS scene
```

`install.sh` does, in order:

1. checks `python3`, `flatpak` and that `com.obsproject.Studio` is installed;
2. copies the CLI to `~/streamrec/streamrec` (`chmod +x`);
3. creates `~/streamrec/.env` from `.env.example` — an **existing `.env` is not touched**;
4. creates the venv `~/streamrec/venv` and installs `requirements.txt` into it;
5. installs `~/.config/systemd/user/streamrec-watch.{service,timer}`
   and `~/.config/autostart/obs-streamrec.desktop` (with `@HOME@` expanded);
6. expands the `@OUTDIR@` placeholder in `basic.ini` and the `YOUR_CHANNEL` placeholder in
   the scene, and copies the OBS profile/scene into
   `~/.var/app/com.obsproject.Studio/config/obs-studio/`.
   **Existing profile/scene files are not overwritten without `--force`.**
7. `systemctl --user daemon-reload && systemctl --user enable --now streamrec-watch.timer`;
8. creates the recordings folder from `STREAMREC_OUTDIR` (default `~/Videos/streams`).

After the first run, fill in `~/streamrec/.env` (see below). If the channel was still empty
when the OBS scene was installed, re-run `./install.sh --force` (this rewrites `.env` too —
keep a copy of your values) or edit the scene name/URLs in OBS by hand.

Then, in OBS itself:

1. **Tools → WebSocket Server Settings**: enable the WebSocket server, port `4455`,
   set a password, and put the same password into `OBS_WS_PASSWORD` in `.env`.
2. Make sure the **scene collection `streamrec`** and the **profile `streamrec`** are selected,
   and the scene (named after your channel) is the active one.
3. The browser sources `Browser` (player) and `Browser 2` (chat) are rewritten by the script
   on every start; you do not have to edit their URLs yourself.

Log out and back in (or run `gtk-launch obs-streamrec`) once, so the autostart entry is picked up.

## Configuration — `~/streamrec/.env`

| Variable | Meaning |
|---|---|
| `STREAMREC_CHANNEL` | **Required.** Twitch channel login to record. The *only* place the channel is configured. Empty → `Ошибка: канал не задан: укажите STREAMREC_CHANNEL ...` |
| `OBS_WS_PASSWORD` | obs-websocket password. `OBS_WEBSOCKET_PASSWORD` is accepted as an alias. |
| `STREAMREC_OUTDIR` | Where recordings are written. Default `~/Videos/streams`. Must match `RecFilePath` in `basic.ini` — `install.sh` fills the profile from this value. |
| `OBS_WS_HOST` | Optional, default `127.0.0.1`. |
| `OBS_WS_PORT` | Optional, default `4455`. |

Real environment variables take precedence over `.env`; `~/streamrec/.env` is read by the
script itself (mode `0600`). `STREAMREC_CHANNEL` accepts `@login` / uppercase and is
normalised to a lowercase login; anything that is not `[A-Za-z0-9_]{1,25}` is rejected.

`Client-ID` for the Twitch GQL liveness check (`kimne78kx3ncx6brgo4mv6wki5h1ko`) is the
**public web client id** that any browser uses on twitch.tv. It is not a secret, it is not
tied to an account, and it is left in the script on purpose; there is no API key or token here.

## Commands

```bash
~/streamrec/streamrec start     # start recording (launches OBS and waits for the websocket if needed)
~/streamrec/streamrec stop      # stop recording and print the file path
~/streamrec/streamrec status    # is it recording, for how long, is the channel live
~/streamrec/streamrec watch     # one check, for the timer: auto start / auto stop
~/streamrec/streamrec launch    # just bring OBS up and wait for the websocket
```

`start --auto` records the source as `auto` in `state.json` — such a recording **is**
stopped automatically when the channel goes offline; a plain `start` is marked `manual`
and the watcher leaves it alone.

State and logs live in `~/streamrec/` (`state.json`, `streamrec.log`).

## Files and layout

```
~/Videos/streams/            # STREAMREC_OUTDIR — Hybrid MP4, "YYYY-MM-DD HH-MM-SS.mp4"
~/streamrec/streamrec        # CLI (this repo's recorder/streamrec)
~/streamrec/.env             # channel, password, output dir (0600)
~/streamrec/venv/            # obsws-python
~/streamrec/state.json       # last file, offline counter, who started the recording
~/streamrec/streamrec.log    # everything the CLI logs
~/.config/systemd/user/streamrec-watch.{service,timer}
~/.config/autostart/obs-streamrec.desktop
~/.var/app/com.obsproject.Studio/config/obs-studio/basic/profiles/streamrec/{basic.ini,recordEncoder.json}
~/.var/app/com.obsproject.Studio/config/obs-studio/basic/scenes/streamrec.json
```

What the profile/scene set up: 1920×1080 @ 60 fps, NVENC H.264 CBR 12 Mbit/s, Hybrid MP4,
AAC 160 kbit/s, scene = `player.twitch.tv/?channel=<channel>&parent=streamlink&quality=chunked`
full-screen + chat popout overlay; the script rewrites the channel into both source URLs.

## Bot integration

See [`examples/telegram-bot-tools.md`](examples/telegram-bot-tools.md) — an anonymised
version of the "recording" section of an agent-bot tool description. The bot only needs to
call the four commands above; no Twitch API key or OAuth token is involved.

## Troubleshooting

- **`pkill -f com.obsproject.Studio` over ssh kills your own ssh session**, because the
  pattern matches the ssh command line itself. Kill by PID, or use `pkill -x obs` (the
  Flatpak binary is named `obs`), or `flatpak kill com.obsproject.Studio`.
  The script's own `kill_obs()` is safe: its `pkill` runs as a separate process whose command
  line is exactly the argument list, and the fallbacks (`pkill -x obs`, `flatpak kill`) cover
  the case where the wrapper is already gone.
- **OBS hangs on the "Safe Mode" dialog after an unclean shutdown** — the websocket never comes
  up, and `--disable-shutdown-check` does not suppress the dialog. The script removes the stale
  `run_*` marks from
  `~/.var/app/com.obsproject.Studio/config/obs-studio/.sentinel/` before launching, and if the
  process is up but the websocket is silent for 60 s it kills and relaunches OBS once — but only
  if no recording file changed in the last 30 s (otherwise the recording would be lost).
- **`stop` waits up to 30 s for `output_active=False`** (OBS keeps flushing the file). If it
  times out, `stop` prints `Остановка запрошена, OBS ещё завершает запись` and exits 1; a
  `status` immediately after may still say `Запись: идёт`.
- **`ffprobe` exists only inside the Flatpak**:
  `flatpak run --command=ffprobe com.obsproject.Studio <file>`.
- **obs-websocket listens on all interfaces.** Always set a password and close port 4455 with a
  firewall; do not expose it to the internet.
- **`status` during a recording shows the previous file** — the path of the current recording is
  only known after `stop`, so `status` prints the last finished file from `state.json`.
- **`start` complains about obs-websocket** → OBS did not come up; look at
  `~/streamrec/streamrec.log`.
- **`Ошибка: канал не задан`** → `STREAMREC_CHANNEL` is empty. Fill in `~/streamrec/.env`.
- **Manual `~/streamrec/streamrec watch` is not affected by the timer** — both use the same
  `state.json`, so run them one at a time if you are debugging.

## Проверка / Troubleshooting (по-русски)

- **`pkill -f com.obsproject.Studio` из ssh убивает сам ssh-сеанс** — шаблон совпадает с
  командной строкой ssh. Гасите по PID, либо `pkill -x obs` (бинарь Flatpak называется `obs`),
  либо `flatpak kill com.obsproject.Studio`. Собственный `kill_obs()` скрипта безопасен: его
  `pkill` — отдельный процесс, а запасные варианты (`pkill -x obs`, `flatpak kill`) добивают
  OBS, когда flatpak-обёртка уже вышла.
- **После некорректного завершения OBS висит на диалоге Safe Mode** — websocket не поднимается,
  `--disable-shutdown-check` диалог не подавляет. Скрипт чистит устаревшие метки `run_*` в
  `~/.var/app/com.obsproject.Studio/config/obs-studio/.sentinel/` перед запуском, а если процесс
  есть, но websocket молчит 60 с — один раз перезапускает OBS, и только если файл записи не
  менялся последние 30 с (иначе запись можно потерять).
- **`stop` ждёт `output_active=False` до 30 с** — OBS дописывает файл. Если не дождался, печатает
  `Остановка запрошена, OBS ещё завершает запись` и возвращает код 1; `status` сразу после `stop`
  может ещё показывать «идёт».
- **`ffprobe` есть только внутри flatpak**:
  `flatpak run --command=ffprobe com.obsproject.Studio <файл>`.
- **obs-websocket слушает все интерфейсы** — обязательно задайте пароль и закройте порт 4455
  файрволом, не выставляйте его в интернет.
- **`status` во время записи показывает прошлый файл** — путь текущей записи известен только
  после `stop`, поэтому `status` печатает последний завершённый файл из `state.json`.
- **`start` ругается на obs-websocket** — OBS не поднялся, смотрите `~/streamrec/streamrec.log`.
- **`Ошибка: канал не задан`** — пуст `STREAMREC_CHANNEL`; заполните `~/streamrec/.env`.
- **Ручной `watch` и таймер делят один `state.json`** — при отладке не запускайте их одновременно.

## License

AGPL-3.0, same as the rest of this repository.
