#!/usr/bin/env bash
# streamrec installer — раскладывает файлы по местам и включает таймер.
#
#   ./install.sh [--force]
#
# Что делает:
#   1. проверяет зависимости (OBS Flatpak, python3, systemd);
#   2. копирует CLI в ~/streamrec/streamrec и ставит ему +x;
#   3. создаёт ~/streamrec/.env из .env.example (существующий .env не трогает);
#   4. создаёт venv ~/streamrec/venv и ставит requirements.txt;
#   5. ставит юниты в ~/.config/systemd/user/ и autostart-файл;
#   6. кладёт профиль и сцену OBS во flatpak-конфиг, подставляя канал и папку записей;
#   7. daemon-reload + enable --now таймера.
#
# --force перезаписывает существующие .env, профиль и сцену OBS.
# Без него установка их не трогает (ваши настройки остаются как есть).

set -euo pipefail

FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force|-f) FORCE=1 ;;
    --help|-h) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Неизвестный аргумент: $arg (см. --help)" >&2; exit 2 ;;
  esac
done

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOME_DIR="${HOME:?HOME не задан}"
DEST="$HOME_DIR/streamrec"
UNIT_DIR="$HOME_DIR/.config/systemd/user"
AUTOSTART_DIR="$HOME_DIR/.config/autostart"
OBS_ROOT="$HOME_DIR/.var/app/com.obsproject.Studio/config/obs-studio"
OBS_PROFILE_DIR="$OBS_ROOT/basic/profiles/streamrec"
OBS_SCENES_DIR="$OBS_ROOT/basic/scenes"
ENV_FILE="$DEST/.env"

say()  { printf '==> %s\n' "$*"; }
warn() { printf 'ВНИМАНИЕ: %s\n' "$*" >&2; }
die()  { printf 'ОШИБКА: %s\n' "$*" >&2; exit 1; }

command -v python3 >/dev/null 2>&1 || die "нет python3"
command -v flatpak >/dev/null 2>&1 || warn "flatpak не найден — OBS запускать будет нечем"
if command -v flatpak >/dev/null 2>&1 && ! flatpak info com.obsproject.Studio >/dev/null 2>&1; then
  warn "flatpak-приложение com.obsproject.Studio не установлено: flatpak install flathub com.obsproject.Studio"
fi

mkdir -p "$DEST" "$UNIT_DIR" "$AUTOSTART_DIR"

# ------------------------------------------------------------------ 1. CLI
say "CLI -> $DEST/streamrec"
install -m 0755 "$SRC/streamrec" "$DEST/streamrec"

# ------------------------------------------------------------------ 2. .env
if [ -f "$ENV_FILE" ] && [ "$FORCE" -ne 1 ]; then
  say ".env уже есть — не трогаю ($ENV_FILE)"
else
  say ".env -> $ENV_FILE (из .env.example)"
  install -m 0600 "$SRC/.env.example" "$ENV_FILE"
fi

# читаем настройки из .env (простые KEY=VALUE без кавычек и подстановок)
env_get() {
  local key="$1" line
  line="$(grep -E "^[[:space:]]*${key}[[:space:]]*=" "$ENV_FILE" 2>/dev/null | tail -n 1 || true)"
  [ -n "$line" ] || return 0
  printf '%s' "${line#*=}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' \
    -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//"
}
CHANNEL="$(env_get STREAMREC_CHANNEL)"
OUTDIR="$(env_get STREAMREC_OUTDIR)"
[ -n "$OUTDIR" ] || OUTDIR="$HOME_DIR/Videos/streams"
case "$OUTDIR" in
  "~"/*) OUTDIR="$HOME_DIR/${OUTDIR#\~/}" ;;
  "~")   OUTDIR="$HOME_DIR" ;;
esac
mkdir -p "$OUTDIR"
say "канал: ${CHANNEL:-<пусто>}; папка записей: $OUTDIR"

# ------------------------------------------------------------------ 3. venv
if [ ! -x "$DEST/venv/bin/python" ]; then
  say "создаю venv -> $DEST/venv"
  if ! python3 -m venv "$DEST/venv"; then
    warn "python3 -m venv не сработал, пробую --system-site-packages"
    python3 -m venv --system-site-packages "$DEST/venv" \
      || die "не удалось создать venv (поставьте python3-venv)"
  fi
fi
say "ставлю зависимости в venv"
"$DEST/venv/bin/python" -m pip install --quiet --upgrade pip
"$DEST/venv/bin/python" -m pip install --quiet -r "$SRC/requirements.txt"

# --------------------------------------------------------------- 4. systemd
say "юниты -> $UNIT_DIR"
install -m 0644 "$SRC/systemd/streamrec-watch.service" "$UNIT_DIR/streamrec-watch.service"
install -m 0644 "$SRC/systemd/streamrec-watch.timer"   "$UNIT_DIR/streamrec-watch.timer"

# -------------------------------------------------------------- 5. autostart
say "autostart -> $AUTOSTART_DIR/obs-streamrec.desktop"
sed "s|@HOME@|$HOME_DIR|g" "$SRC/autostart/obs-streamrec.desktop" \
  > "$AUTOSTART_DIR/obs-streamrec.desktop"
chmod 0644 "$AUTOSTART_DIR/obs-streamrec.desktop"

# ---------------------------------------------------------- 6. OBS: профиль
install_obs_file() {
  local src="$1" dst="$2"
  if [ -e "$dst" ] && [ "$FORCE" -ne 1 ]; then
    warn "уже существует, не трогаю: $dst (перезаписать — install.sh --force)"
    return 0
  fi
  mkdir -p "$(dirname "$dst")"
  say "OBS-файл -> $dst"
  sed "s|@OUTDIR@|$OUTDIR|g" "$src" > "$dst"
}

install_obs_file "$SRC/obs/profile/basic.ini"              "$OBS_PROFILE_DIR/basic.ini"
install_obs_file "$SRC/obs/profile/recordEncoder.json"     "$OBS_PROFILE_DIR/recordEncoder.json"
install_obs_file "$SRC/obs/scenes/streamrec.json"          "$OBS_SCENES_DIR/streamrec.json"

# в сцене канал — плейсхолдер YOUR_CHANNEL; подставляем только что установленный файл
if grep -q "YOUR_CHANNEL" "$OBS_SCENES_DIR/streamrec.json" 2>/dev/null; then
  if [ -n "$CHANNEL" ]; then
    sed -i "s|YOUR_CHANNEL|$CHANNEL|g" "$OBS_SCENES_DIR/streamrec.json"
    say "в сцене OBS канал -> $CHANNEL"
  else
    warn "STREAMREC_CHANNEL пуст: в сцене OBS остался плейсхолдер YOUR_CHANNEL."
    warn "Заполните $ENV_FILE и повторите install.sh --force (или поправьте сцену руками)."
  fi
fi

# ---------------------------------------------------------------- 7. таймер
if command -v systemctl >/dev/null 2>&1; then
  say "включаю таймер streamrec-watch.timer"
  systemctl --user daemon-reload || warn "systemctl --user daemon-reload не сработал"
  systemctl --user enable --now streamrec-watch.timer \
    || warn "не удалось включить таймер (нет пользовательской сессии systemd?)"
else
  warn "systemctl не найден — таймер не включён, запускайте 'streamrec watch' из cron"
fi

# ------------------------------------------------------------------ итог
echo
say "готово."
if [ -z "$CHANNEL" ]; then
  echo "  Осталось одно: впишите канал и пароль obs-websocket в $ENV_FILE"
  echo "    строка STREAMREC_CHANNEL — логин канала на Twitch"
  echo "    строка OBS_WS_PASSWORD — пароль из Tools -> WebSocket Server Settings"
  echo "  затем: \"$DEST/streamrec\" status"
else
  echo "  Проверка: \"$DEST/streamrec\" status"
fi
echo "  OBS: профиль 'streamrec' и сцена '$CHANNEL' (Scene Collection 'streamrec')."
echo "  obs-websocket: Tools -> WebSocket Server Settings, порт 4455, пароль — в .env."
echo "  Логи: $DEST/streamrec.log"
