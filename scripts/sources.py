# -*- coding: utf-8 -*-
"""Единый поиск исходного видео по имени.

    from sources import find_source
    find_source("game01")   -> <stream_root>/game01.mov
    find_source("game02")   -> <stream_root>/<game>/game02.mp4

Исходники лежат либо в корне `stream_root` (tps1.mov, gasstation1.mp4, ...), либо
в подпапке игры ровно на один уровень вложенности (<game>/<game>1.mp4).
Порядок поиска: сначала корень по порядку расширений (прежнее поведение), затем
подпапки. Служебные подпапки (_*, clips*, загрузить) в поиске не участвуют.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG  # noqa: E402

EXTS = (".mov", ".mp4", ".mkv", ".webm")
SKIP_PREFIXES = ("_", "clips")
SKIP_DIRS = ("загрузить",)


def _skip_dir(dirname: str) -> bool:
    low = dirname.lower()
    if dirname in SKIP_DIRS or low in [d.lower() for d in SKIP_DIRS]:
        return True
    return any(low.startswith(p) for p in SKIP_PREFIXES)


def _subdirs() -> list:
    try:
        entries = sorted(os.scandir(CONFIG.stream_root), key=lambda e: e.name)
    except OSError:
        return []
    return [e.path for e in entries if e.is_dir() and not _skip_dir(e.name)]


def find_source(name: str) -> str:
    """Абсолютный путь к исходнику <name> (без расширения).

    FileNotFoundError, если файла нет; FileNotFoundError со списком найденных,
    если в подпапках нашлось больше одного кандидата.
    """
    for ext in EXTS:
        root_file = os.path.join(CONFIG.stream_root, name + ext)
        if os.path.isfile(root_file):
            return root_file

    found: list = []
    for sub in _subdirs():
        for ext in EXTS:
            p = os.path.join(sub, name + ext)
            if os.path.isfile(p):
                found.append(p)
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise FileNotFoundError(
            "видео %s найдено больше одного раза: %s" % (name, "; ".join(sorted(found)))
        )
    raise FileNotFoundError("нет видео %s в %s и подпапках" % (name, CONFIG.stream_root))
