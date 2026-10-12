"""Download the official LTO Filipino Driver's Manual PDFs into data/raw/.

Usage:  python scripts/download_docs.py
If your network blocks the download, open the URLs printed below in a
browser and save the PDFs into data/raw/ with the same file names.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from driveready import config  # noqa: E402


def main() -> int:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    failed = 0
    for doc_id, meta in config.SOURCES.items():
        dest = config.RAW_DIR / meta["file"]
        if dest.exists() and dest.stat().st_size > 10_000:
            print(f"[ok]   {dest.name} already present")
            continue
        print(f"[get]  {meta['url']}")
        try:
            r = requests.get(meta["url"], timeout=120, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                raise ValueError("response is not a PDF")
            dest.write_bytes(r.content)
            print(f"[ok]   saved {dest.name} ({len(r.content) / 1e6:.1f} MB)")
        except Exception as e:  # noqa: BLE001 - report any failure and continue
            failed += 1
            print(f"[fail] {doc_id}: {e}\n       Download it manually and save as data/raw/{meta['file']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
