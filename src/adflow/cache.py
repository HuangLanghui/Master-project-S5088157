"""On-disk cache for the expensive, deterministic stages.

Background generation is 99% of a grid run: 27s per format against 0.2s for
everything else combined. Most of that work is repeated. A background depends
on the prompt, the format size, the seed and the backend, but not on the
relighting applied afterwards, so a sweep over five relight levels regenerates
the same five images. Extraction is likewise independent of seed, backend and
relight, yet runs once per pipeline call.

Caching both turns a 7.5-hour grid into roughly 1.5 hours with identical
output, since both stages are deterministic given their inputs.

The key is a digest of everything that feeds the result. Any change to the
prompt, size, seed or backend produces a different key rather than a stale hit,
so there is no invalidation to remember. Entries are plain PNGs and small JSON
sidecars, which makes a partial run resumable and lets a suspect entry be
inspected or deleted by hand.

Disable with ADFLOW_NO_CACHE=1 when timing a cold run.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .utils import log

CACHE_DIR = Path(os.environ.get("ADFLOW_CACHE_DIR", ".cache"))


def enabled() -> bool:
    return os.environ.get("ADFLOW_NO_CACHE", "") not in ("1", "true", "yes")


def digest(*parts) -> str:
    payload = "\x1f".join(repr(p) for p in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def file_identity(path: str | Path) -> tuple:
    """Identify a source file without reading it.

    Size and mtime are enough here: the dataset is static during a run, and a
    file that is edited mid-run gets a new mtime and therefore a new key.
    """
    p = Path(path)
    try:
        st = p.stat()
        return (p.name, st.st_size, int(st.st_mtime))
    except OSError:
        return (str(p), 0, 0)


def image(namespace: str, key: str, produce: Callable[[], Image.Image]) -> Image.Image:
    """Return a cached image, producing and storing it on a miss."""
    if not enabled():
        return produce()

    path = CACHE_DIR / namespace / f"{key}.png"
    if path.exists():
        with Image.open(path) as cached:
            return cached.copy()

    result = produce()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary name first so an interrupted run cannot leave a
    # truncated PNG that later reads as a valid cache hit.
    tmp = path.with_suffix(".part")
    result.save(tmp, format="PNG")
    tmp.replace(path)
    return result


def image_with_meta(namespace: str, key: str,
                    produce: Callable[[], tuple[Image.Image, dict]],
                    ) -> tuple[Image.Image, dict]:
    """As `image`, for producers that also return a small metadata dict."""
    if not enabled():
        return produce()

    png = CACHE_DIR / namespace / f"{key}.png"
    meta = CACHE_DIR / namespace / f"{key}.json"
    if png.exists() and meta.exists():
        with Image.open(png) as cached:
            return cached.copy(), json.loads(meta.read_text(encoding="utf-8"))

    result, payload = produce()
    png.parent.mkdir(parents=True, exist_ok=True)
    tmp = png.with_suffix(".part")
    result.save(tmp, format="PNG")
    tmp.replace(png)
    meta.write_text(json.dumps(payload), encoding="utf-8")
    return result, payload


def clear(namespace: str | None = None) -> int:
    """Delete cached entries. Returns the number of files removed."""
    root = CACHE_DIR / namespace if namespace else CACHE_DIR
    if not root.exists():
        return 0
    removed = 0
    for p in root.rglob("*"):
        if p.is_file():
            p.unlink()
            removed += 1
    log.info("cache: removed %d file(s) from %s", removed, root)
    return removed


def stats() -> dict:
    """Entry count and total size per namespace, for reporting a run."""
    out: dict[str, dict] = {}
    if not CACHE_DIR.exists():
        return out
    for ns in sorted(p for p in CACHE_DIR.iterdir() if p.is_dir()):
        files = [f for f in ns.iterdir() if f.suffix == ".png"]
        out[ns.name] = {
            "entries": len(files),
            "megabytes": round(sum(f.stat().st_size for f in files) / 1e6, 1),
        }
    return out
