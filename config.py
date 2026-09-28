# -*- coding: utf-8 -*-
"""Настройки конвейера: загрузка `config.json`.

Путь к конфигу ищется так:
  1. `$STS_CONFIG` — если переменная задана, берётся ровно этот файл;
  2. иначе `config.json` в корне репозитория (рядом с этим файлом).

`config.example.json` — шаблон: скопируйте его в `config.json` и поправьте пути
(`config.json` в `.gitignore`). Если файла нет — понятная ошибка, а не падение
где-то в середине рендера.

Использование:
    from config import CONFIG, require_path, repo_path
    CONFIG.stream_root            # корень с исходными записями
    CONFIG.path("badwords_path")  # путь из конфига (строка)
"""
from __future__ import annotations

import json
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "config.json")

MISSING = (
    "нет файла настроек: {cfg}\n"
    "скопируйте config.example.json в config.json и поправьте пути\n"
    "(или укажите путь к своему конфигу в переменной окружения STS_CONFIG)"
)

REQUIRED = (
    "в конфиге {cfg} нет ключа {key!r} (см. config.example.json)"
)


class Config:
    """Значения из `config.json` как атрибуты + пути по умолчанию внутри репо."""

    def __init__(self, data: dict, path: str) -> None:
        self._data = data
        self.config_path = path
        self.stream_root = os.path.abspath(os.path.expanduser(data["stream_root"]))
        self.scripts_dir = os.path.join(REPO_ROOT, "scripts")
        self.assets_dir = os.path.join(REPO_ROOT, "assets")
        self.out_dir = self._under_repo(data.get("out_dir"), "out")
        self.temp_dir = self._under_repo(data.get("temp_dir"), os.path.join("_cache", "temp"))
        self.layouts = data.get("layouts") or {}
        self.boom_wav = os.path.join(self.assets_dir, "boom.wav")

    # ------------------------------------------------------------------ служебное
    def _under_repo(self, value, default: str) -> str:
        """Значение из конфига; относительный путь — от корня репозитория."""
        if not value:
            return os.path.join(REPO_ROOT, default)
        value = os.path.expanduser(str(value))
        if not os.path.isabs(value):
            value = os.path.join(REPO_ROOT, value)
        return os.path.normpath(value)

    def _required(self, key: str) -> object:
        if key not in self._data:
            raise SystemExit(REQUIRED.format(path=self.config_path, key=key))
        return self._data[key]

    def path(self, key: str) -> str:
        """Путь из конфига (обязательный ключ), раскрытый до абсолютного."""
        value = str(self._required(key)).strip()
        return os.path.abspath(os.path.expanduser(value))

    # ------------------------------------------------------------------ папки
    @property
    def analysis_dir(self) -> str:
        """Где лежат результаты анализа: <stream_root>/_analysis (по умолчанию)."""
        custom = self._data.get("analysis_dir")
        if custom:
            return os.path.abspath(os.path.expanduser(str(custom)))
        return os.path.join(self.stream_root, "_analysis")

    @property
    def features_dir(self) -> str:
        """Где искать features.csv: по умолчанию там же, где анализ."""
        custom = self._data.get("features_dir")
        if custom:
            return os.path.abspath(os.path.expanduser(str(custom)))
        return self.analysis_dir

    @property
    def clips_root(self) -> str:
        """Готовые нарезки для `scripts/clips/render_vertical.py`."""
        custom = self._data.get("clips_root")
        if custom:
            return os.path.abspath(os.path.expanduser(str(custom)))
        return os.path.join(self.stream_root, "clips")

    @property
    def clips_vertical_root(self) -> str:
        custom = self._data.get("clips_vertical_root")
        if custom:
            return os.path.abspath(os.path.expanduser(str(custom)))
        return os.path.join(self.stream_root, "clips_vertical")

    # ------------------------------------------------------------------ словари
    @property
    def badwords_path(self) -> str:
        return self.path("badwords_path")

    @property
    def okwords_path(self) -> str:
        return self.path("okwords_path")

    # ------------------------------------------------------------------ шрифты
    @property
    def fonts_dir(self) -> str:
        return self.path("fonts_dir")

    @property
    def font_caption(self) -> str:
        """Шрифт субтитров (ASS-стиль и перенос строк)."""
        return self.path("font_caption")

    @property
    def font_title(self) -> str:
        """Шрифт плашки-заголовка вертикалки."""
        return self.path("font_title")

    @property
    def font_intro(self) -> str:
        """Шрифт интро-дисклеймера."""
        return self.path("font_intro")

    @property
    def font_intro_body(self) -> str:
        """Шрифт тела текста интро; по умолчанию — как `font_intro`."""
        value = self._data.get("font_intro_body")
        return self.path("font_intro_body") if value else self.font_intro

    @property
    def font_intro(self) -> str:
        return self.path("font_intro")

    # ------------------------------------------------------------------ инструменты
    @property
    def ffmpeg(self) -> str:
        return self.path("ffmpeg")

    @property
    def ffprobe(self) -> str:
        return self.path("ffprobe")

    @property
    def separator_exe(self) -> str:
        return self.path("separator_exe")

    @property
    def separator_model_dir(self) -> str:
        return self.path("separator_model_dir")

    @property
    def separator_model(self) -> str:
        return str(self._data.get("separator_model", "vocals_mel_band_roformer.ckpt"))

    # ------------------------------------------------------------------ цензура
    def censor(self) -> dict:
        """Правило цензуры; дефолты — как в исходных скриптах."""
        rule = dict(self._data.get("censor") or {})
        rule.setdefault("window_frac", 0.40)     # доля длины слова
        rule.setdefault("min_window_sec", 0.15)  # минимум, с
        rule.setdefault("word_pad_sec", 0.03)    # паддинг для целых слов (пидор*)
        return rule


def load(path: str | None = None) -> Config:
    """Прочитать конфиг. Без аргумента — `$STS_CONFIG` или `config.json` в корне."""
    cfg_path = path or os.environ.get("STS_CONFIG") or DEFAULT_CONFIG_PATH
    cfg_path = os.path.abspath(os.path.expanduser(cfg_path))
    if not os.path.isfile(cfg_path):
        raise SystemExit(MISSING.format(cfg=cfg_path))
    with open(cfg_path, encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except ValueError as exc:
            raise SystemExit("конфиг %s — не JSON: %s" % (cfg_path, exc))
    if not isinstance(data, dict):
        raise SystemExit("конфиг %s: ожидался объект JSON" % cfg_path)
    if "stream_root" not in data:
        raise SystemExit(REQUIRED.format(cfg=cfg_path, key="stream_root"))
    return Config(data, cfg_path)


CONFIG = load()


def path_of(key: str) -> str:
    return CONFIG.path(key)


def require_path(key: str, what: str, hint: str = "") -> str:
    """Путь из конфига + проверка существования (понятная ошибка, а не IOError)."""
    value = CONFIG.path(key)
    if not os.path.exists(value):
        tail = ("\n" + hint) if hint else ""
        raise SystemExit("нет %s: %s (ключ %r в %s)%s"
                         % (what, value, key, CONFIG.config_path, tail))
    return value


def repo_path(*parts: str) -> str:
    """Путь внутри репозитория (относительно config.py)."""
    return os.path.join(REPO_ROOT, *parts)


def analysis_dir_for(src: str) -> str:
    """Папка анализа одного исходника: <analysis_dir>/<src>."""
    return os.path.join(CONFIG.analysis_dir, src)
