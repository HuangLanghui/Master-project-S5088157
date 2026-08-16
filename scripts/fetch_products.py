#!/usr/bin/env python3
"""Collect candidate product images, recording provenance as they download.

Each API response carries the licence, the author and the origin URL, so
provenance is stored with the file rather than reconstructed later.

Sources:

    pexels     stock library, clean pack shots, permissive licence, no
               attribution required. Needs a free key in PEXELS_API_KEY.
    unsplash   stock library, higher artistic quality but fewer plain pack
               shots. Needs a free key in UNSPLASH_ACCESS_KEY.
    commons    no key required and good provenance, but it is an encyclopedic
               archive rather than a stock library. Search relevance for pack
               shots is poor (about eight usable images out of fifty-two on a
               trial run) and it holds historical material unsuitable for a
               modern dataset. Review everything it returns.

    export PEXELS_API_KEY=...            # PowerShell: $env:PEXELS_API_KEY="..."
    python scripts/fetch_products.py --source pexels --limit 8
    python scripts/fetch_products.py --source pexels --query "matte coffee pouch"

This writes candidates, not a dataset. Review them and promote the keepers
into data/manifest.json.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia asks for a descriptive User-Agent identifying the tool and purpose.
UA = "adflow-thesis/0.2 (academic dataset collection; https://commons.wikimedia.org)"

# Searches chosen to fill the crossed quotas in data/README.md. The trailing
# terms bias Commons toward isolated pack shots rather than shelf photography.
QUERIES: dict[str, str] = {
    "coffee_bag": "coffee bag packaging product",
    "cereal_carton": "cereal box packaging",
    "book_cover": "hardcover book white background",
    "skincare_tube": "cosmetic tube product",
    "canvas_pack": "backpack product white background",
    "ceramic_mug": "ceramic mug product white background",
    "aluminium_can": "aluminium beverage can white background",
    "wine_bottle": "wine bottle white background",
    "steel_flask": "stainless steel vacuum flask",
    "sunglasses": "sunglasses white background",
    "headphones": "headphones product white background",
    "sneaker": "sneaker shoe white background",
    "candle_jar": "scented candle jar",
    "camera": "camera product white background",
    "smartphone": "smartphone product white background",
    "watch": "wristwatch dial close up",
    "earbud_case": "wireless earbuds charging case",
    "perfume": "perfume bottle product",
}

_TAGS = re.compile(r"<[^>]+>")


def _clean(value: str) -> str:
    """Commons returns HTML fragments in extmetadata; flatten them."""
    return html.unescape(_TAGS.sub("", value or "")).strip()


def _get(url: str, headers: dict[str, str] | None = None) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read())


def _require_key(env: str, where: str) -> str:
    key = os.environ.get(env, "").strip()
    if not key:
        raise SystemExit(
            f"{env} is not set. Get a free key at {where}, then:\n"
            f"  PowerShell:  $env:{env}=\"...\"\n"
            f"  bash:        export {env}=...\n"
            "Do not paste the key into this file - it would be committed to git."
        )
    return key


def search_pexels(query: str, limit: int, min_width: int) -> list[dict]:
    key = _require_key("PEXELS_API_KEY", "https://www.pexels.com/api/new/")
    url = (f"https://api.pexels.com/v1/search?query={urllib.parse.quote(query)}"
           f"&per_page={min(80, limit * 3)}")
    photos = _get(url, {"Authorization": key}).get("photos", []) or []

    out: list[dict] = []
    for p in photos:
        if min(p.get("width", 0), p.get("height", 0)) < min_width:
            continue
        out.append({
            "title": p.get("alt") or f"pexels-{p.get('id')}",
            # `large2x` is ~1880px on the long edge: plenty for a 1080px canvas
            # and a fraction of the original's weight.
            "url": (p.get("src") or {}).get("large2x") or (p.get("src") or {}).get("original"),
            "source": p.get("url", ""),
            "licence": "Pexels License",
            "author": p.get("photographer", ""),
            "width": p.get("width", 0),
            "height": p.get("height", 0),
        })
        if len(out) >= limit:
            break
    return out


def search_unsplash(query: str, limit: int, min_width: int) -> list[dict]:
    key = _require_key("UNSPLASH_ACCESS_KEY", "https://unsplash.com/oauth/applications")
    url = (f"https://api.unsplash.com/search/photos?query={urllib.parse.quote(query)}"
           f"&per_page={min(30, limit * 3)}&content_filter=high")
    results = _get(url, {"Authorization": f"Client-ID {key}"}).get("results", []) or []

    out: list[dict] = []
    for p in results:
        if min(p.get("width", 0), p.get("height", 0)) < min_width:
            continue
        out.append({
            "title": p.get("alt_description") or f"unsplash-{p.get('id')}",
            "url": (p.get("urls") or {}).get("regular"),
            "source": (p.get("links") or {}).get("html", ""),
            "licence": "Unsplash License",
            "author": (p.get("user") or {}).get("name", ""),
            "width": p.get("width", 0),
            "height": p.get("height", 0),
            # Unsplash's API terms require pinging this endpoint whenever an
            # image is actually downloaded. It is a tracking hook, not the file.
            "_download_ping": (p.get("links") or {}).get("download_location", ""),
        })
        if len(out) >= limit:
            break
    return out


def search_commons(query: str, limit: int, min_width: int) -> list[dict]:
    url = (f"{API}?action=query&format=json&generator=search"
           f"&gsrsearch={urllib.parse.quote(query + ' filetype:bitmap')}"
           f"&gsrnamespace=6&gsrlimit={limit * 3}"
           f"&prop=imageinfo&iiprop=url|size|extmetadata&iiurlwidth=1600")
    pages = (_get(url).get("query", {}) or {}).get("pages", {}) or {}

    out: list[dict] = []
    for page in pages.values():
        info = page.get("imageinfo", [{}])[0]
        meta = info.get("extmetadata", {}) or {}
        width, height = info.get("width", 0), info.get("height", 0)
        if min(width, height) < min_width:
            continue
        # Extreme panoramas are never single-product pack shots.
        if max(width, height) / max(1, min(width, height)) > 3.0:
            continue
        out.append({
            "title": page["title"],
            "url": info.get("thumburl") or info.get("url"),
            "source": info.get("descriptionurl", ""),
            "licence": _clean(meta.get("LicenseShortName", {}).get("value", "")),
            "author": _clean(meta.get("Artist", {}).get("value", "")),
            "width": width,
            "height": height,
        })
        if len(out) >= limit:
            break
    return out


SOURCES = {
    "pexels": search_pexels,
    "unsplash": search_unsplash,
    "commons": search_commons,
}


def search(source: str, query: str, limit: int, min_width: int) -> list[dict]:
    return SOURCES[source](query, limit, min_width)


def download(entry: dict, dest: Path) -> bool:
    if dest.exists():
        return True

    ping = entry.get("_download_ping")
    if ping:
        # Required by the Unsplash API guidelines; failure to report a download
        # is a terms violation, but it must not abort the collection run.
        try:
            key = os.environ.get("UNSPLASH_ACCESS_KEY", "")
            _get(ping, {"Authorization": f"Client-ID {key}"})
        except Exception:  # noqa: BLE001
            pass

    req = urllib.request.Request(entry["url"], headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = resp.read()
    except Exception as exc:  # noqa: BLE001
        print(f"    ! {dest.name}: {exc}")
        return False
    dest.write_bytes(data)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=sorted(SOURCES), default="pexels")
    ap.add_argument("--out", default="data/candidates")
    ap.add_argument("--limit", type=int, default=6, help="candidates per query")
    ap.add_argument("--min-width", type=int, default=800)
    ap.add_argument("--query", help="run one ad-hoc search instead of the built-in set")
    args = ap.parse_args()

    queries = {"adhoc": args.query} if args.query else QUERIES
    out_root = Path(args.out) / args.source
    out_root.mkdir(parents=True, exist_ok=True)

    index: list[dict] = []
    for slug, query in queries.items():
        print(f"{slug}: {query!r}")
        try:
            hits = search(args.source, query, args.limit, args.min_width)
        except Exception as exc:  # noqa: BLE001
            print(f"    ! search failed: {exc}")
            continue

        folder = out_root / slug
        folder.mkdir(exist_ok=True)
        for i, hit in enumerate(hits):
            suffix = Path(urllib.parse.urlparse(hit["url"]).path).suffix or ".jpg"
            dest = folder / f"{slug}_{i:02d}{suffix}"
            if download(hit, dest):
                print(f"    {dest.name}  {hit['width']}x{hit['height']}  {hit['licence']}")
                record = {k: v for k, v in hit.items() if not k.startswith("_")}
                index.append({**record, "source_api": args.source, "slug": slug,
                              "file": str(dest).replace("\\", "/")})
            time.sleep(0.4)  # be polite to the API
        time.sleep(0.4)

    (out_root / "candidates.json").write_text(
        json.dumps({"candidates": index}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(index)} candidate(s) -> {out_root}/candidates.json")
    print("Review them, then copy the keepers into data/products/ and add manifest rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
