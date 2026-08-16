"""Cache correctness. A stale hit would silently corrupt every reported number."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from adflow import cache


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.delenv("ADFLOW_NO_CACHE", raising=False)


def _red():
    return Image.new("RGB", (32, 32), (255, 0, 0))


def _blue():
    return Image.new("RGB", (32, 32), (0, 0, 255))


def test_second_call_does_not_recompute():
    calls = []

    def produce():
        calls.append(1)
        return _red()

    a = cache.image("ns", "k", produce)
    b = cache.image("ns", "k", produce)
    assert len(calls) == 1
    assert np.array_equal(np.array(a), np.array(b))


def test_a_different_key_is_a_different_entry():
    first = cache.image("ns", "k1", _red)
    second = cache.image("ns", "k2", _blue)
    assert not np.array_equal(np.array(first), np.array(second))


def test_digest_changes_with_every_input():
    base = cache.digest("bg", "turbo", "prompt", (1080, 1080), 42)
    assert base != cache.digest("bg", "sdxl", "prompt", (1080, 1080), 42)
    assert base != cache.digest("bg", "turbo", "other", (1080, 1080), 42)
    assert base != cache.digest("bg", "turbo", "prompt", (1080, 1920), 42)
    assert base != cache.digest("bg", "turbo", "prompt", (1080, 1080), 43)
    assert base == cache.digest("bg", "turbo", "prompt", (1080, 1080), 42)


def test_metadata_round_trips_with_the_image():
    def produce():
        return _red(), {"bbox": [1, 2, 3, 4], "method": "rembg"}

    cache.image_with_meta("ns", "k", produce)
    _, meta = cache.image_with_meta("ns", "k", lambda: (_blue(), {"bbox": [], "method": "x"}))
    assert meta == {"bbox": [1, 2, 3, 4], "method": "rembg"}


def test_rgba_alpha_survives_the_round_trip():
    """Extraction stores its mask as the cut-out's alpha channel, so a cache
    that dropped alpha would silently return a fully opaque product."""
    src = Image.new("RGBA", (16, 16), (10, 20, 30, 0))
    src.putpixel((8, 8), (10, 20, 30, 255))

    cache.image("ns", "rgba", lambda: src)
    out = cache.image("ns", "rgba", lambda: Image.new("RGBA", (16, 16)))
    assert np.array_equal(np.array(out)[:, :, 3], np.array(src)[:, :, 3])


def test_disabled_cache_always_recomputes(monkeypatch):
    monkeypatch.setenv("ADFLOW_NO_CACHE", "1")
    calls = []

    def produce():
        calls.append(1)
        return _red()

    cache.image("ns", "k", produce)
    cache.image("ns", "k", produce)
    assert len(calls) == 2


def test_file_identity_tracks_edits(tmp_path):
    f = tmp_path / "a.jpg"
    f.write_bytes(b"one")
    before = cache.file_identity(f)
    f.write_bytes(b"a much longer payload")
    assert cache.file_identity(f) != before


def test_missing_file_does_not_raise():
    assert cache.file_identity("no/such/file.jpg")[1] == 0
