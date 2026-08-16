"""Minimal ComfyUI HTTP client.

Talks to a locally running ComfyUI instance (default http://127.0.0.1:8188)
using the /prompt, /history and /view endpoints. Workflows are stored as
API-format JSON under workflows/ and patched with runtime values
(prompt text, seed, resolution) before submission.
"""
from __future__ import annotations

import io
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

from .utils import log


class ComfyUIClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188", timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -- availability -----------------------------------------------------
    def is_available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/system_stats", timeout=3):
                return True
        except Exception:  # noqa: BLE001
            return False

    # -- workflow execution ----------------------------------------------
    def run_workflow(self, workflow_path: str | Path, patches: dict[str, dict]) -> Image.Image:
        """Submit a workflow and block until its first output image is ready.

        patches: {node_id: {input_name: value}} applied onto the API JSON.
        """
        graph = json.loads(Path(workflow_path).read_text(encoding="utf-8"))
        for node_id, inputs in patches.items():
            graph[node_id]["inputs"].update(inputs)

        req = urllib.request.Request(
            f"{self.base_url}/prompt",
            data=json.dumps({"prompt": graph}).encode(),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            prompt_id = json.loads(resp.read())["prompt_id"]
        log.info("ComfyUI: queued prompt %s", prompt_id)

        deadline = time.time() + self.timeout
        while time.time() < deadline:
            time.sleep(1.5)
            with urllib.request.urlopen(f"{self.base_url}/history/{prompt_id}", timeout=10) as resp:
                hist = json.loads(resp.read())
            if prompt_id in hist:
                return self._first_image(hist[prompt_id])
        raise TimeoutError(f"ComfyUI did not finish within {self.timeout}s")

    def _first_image(self, entry: dict) -> Image.Image:
        for node_out in entry.get("outputs", {}).values():
            for meta in node_out.get("images", []):
                qs = urllib.parse.urlencode({
                    "filename": meta["filename"],
                    "subfolder": meta.get("subfolder", ""),
                    "type": meta.get("type", "output"),
                })
                with urllib.request.urlopen(f"{self.base_url}/view?{qs}", timeout=30) as resp:
                    return Image.open(io.BytesIO(resp.read())).convert("RGB")
        raise RuntimeError("Workflow finished but produced no images")
