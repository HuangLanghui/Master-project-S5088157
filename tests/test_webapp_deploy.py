"""Hosted mode must actually restrict what the hosted instance will attempt.

The free tier has 512 MB and no GPU. One render peaks near 400 MB there, so the
two guards below are not cosmetic: offering the diffusers backend would crash
the process, and an unbounded upload would let a visitor do the same.
"""
from __future__ import annotations

import importlib
import io
import sys
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("fastapi", reason="the harness is behind the `web` extra")

# Guard the import itself rather than the packages behind it. TestClient needs
# an HTTP client that starlette has renamed once already - httpx, then httpx2 -
# and it raises RuntimeError rather than ImportError when it cannot find one,
# which importorskip does not catch. An unguarded import therefore aborts
# collection for the whole suite instead of skipping this module, and naming
# either package here would skip on the version that ships the other.
try:
    from fastapi.testclient import TestClient
except (ImportError, RuntimeError) as exc:                  # pragma: no cover
    pytest.skip(f"no test client: {exc}", allow_module_level=True)

WEBAPP = Path(__file__).resolve().parent.parent / "webapp"
if str(WEBAPP) not in sys.path:
    sys.path.insert(0, str(WEBAPP))

DIFFUSERS_OPTION = '<option value="diffusers"'


def _app(monkeypatch, deployed: bool):
    """Import webapp/app.py fresh, since it reads the flag at import time."""
    monkeypatch.setenv("ADFLOW_DEPLOY", "1" if deployed else "0")
    sys.modules.pop("app", None)
    return importlib.import_module("app")


def test_local_mode_offers_both_backends(monkeypatch):
    app = _app(monkeypatch, deployed=False)
    assert app.DEPLOYED is False
    assert DIFFUSERS_OPTION in TestClient(app.app).get("/").text


def test_deployed_mode_hides_the_gpu_backend(monkeypatch):
    app = _app(monkeypatch, deployed=True)
    assert app.DEPLOYED is True
    body = TestClient(app.app).get("/").text
    assert DIFFUSERS_OPTION not in body, "a GPU backend was offered on a CPU host"
    assert '<option value="procedural"' in body


def test_deployed_mode_rejects_an_oversized_upload(monkeypatch):
    app = _app(monkeypatch, deployed=True)
    monkeypatch.setattr(app, "MAX_UPLOAD_BYTES", 1024)

    blob = io.BytesIO()
    Image.new("RGB", (400, 400), (120, 60, 30)).save(blob, format="PNG")
    assert blob.tell() > 1024, "fixture is not big enough to trip the limit"
    blob.seek(0)

    response = TestClient(app.app).post(
        "/generate",
        data={"product": "", "backend": "procedural", "relight": "0.0",
              "seed": "42", "ablation": "full"},
        files={"upload": ("big.png", blob, "image/png")})

    assert response.status_code == 200
    assert "Upload rejected" in response.text


def test_hosted_page_does_not_advise_on_absent_controls(monkeypatch):
    """No guidance about a backend the hosted form does not offer."""
    hosted = TestClient(_app(monkeypatch, deployed=True).app).get("/").text
    assert "diffusers render" not in hosted
    assert "seconds depending on the canvas" in hosted

    local = TestClient(_app(monkeypatch, deployed=False).app).get("/").text
    assert "diffusers render" in local


def test_deployed_mode_serves_the_precomputed_cutout(monkeypatch):
    """Matting does not fit on the host.

    One extraction peaks at 729 MB through rembg against a 512 MB instance, so
    a dataset product has to reach the pipeline as an alpha channel that was
    computed offline. If this lookup stops resolving, the host silently goes
    back to matting and the first visitor to press Generate kills the process.
    """
    app = _app(monkeypatch, deployed=True)
    resolved = app.precut("data/products/camera_lens.jpeg")
    assert resolved is not None, "no pre-cut input; run scripts/make_cutouts.py"
    assert resolved.exists() and resolved.suffix == ".png"

    from PIL import Image
    with Image.open(resolved) as img:
        assert img.mode == "RGBA"
        assert img.getchannel("A").getextrema()[0] == 0, "no transparency to read"


def test_precut_is_local_only(monkeypatch):
    """Locally the pipeline must still run Stage 2 for real."""
    app = _app(monkeypatch, deployed=False)
    assert app.precut("data/products/camera_lens.jpeg") is None


def test_every_manifest_product_has_a_cutout(monkeypatch):
    app = _app(monkeypatch, deployed=True)
    from adflow.dataset import load_manifest

    missing = [p.id for p in load_manifest()
               if app.precut(p.image) is None]
    assert not missing, f"no pre-cut input for {missing}"


def test_study_survives_a_missing_stimulus_directory(monkeypatch, tmp_path):
    """experiments/samples is gitignored, so a deployment never has it.

    Iterating it unguarded raised FileNotFoundError and served a 500 rather
    than the page's own empty state.
    """
    app = _app(monkeypatch, deployed=True)
    monkeypatch.setattr(app.study, "SAMPLES", tmp_path / "does_not_exist")

    assert app.study.build_session() == []
    response = TestClient(app.app).get("/study")
    assert response.status_code == 200
    assert "Image comparison study" in response.text


def test_health_is_reachable_for_the_platform_probe(monkeypatch):
    app = _app(monkeypatch, deployed=True)
    response = TestClient(app.app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
