"""Download versioned SignTurk model assets and verify their SHA-256 hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "model-assets.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(asset: dict[str, object], force: bool = False) -> None:
    target = ROOT / str(asset["path"])
    expected_hash = str(asset["sha256"])
    expected_size = int(asset["bytes"])

    if target.exists() and not force:
        if target.stat().st_size == expected_size and sha256(target) == expected_hash:
            print(f"verified  {target.relative_to(ROOT)}")
            return
        print(f"replacing {target.relative_to(ROOT)} (checksum mismatch)")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(
        str(asset["url"]), headers={"User-Agent": "SignTurk-model-downloader/1"}
    )
    print(f"download  {asset['name']} -> {target.relative_to(ROOT)}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        if temporary.stat().st_size != expected_size:
            raise RuntimeError(
                f"size mismatch for {asset['name']}: "
                f"expected {expected_size}, got {temporary.stat().st_size}"
            )
        actual_hash = sha256(temporary)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"checksum mismatch for {asset['name']}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", choices=("live", "research", "all"), default="live")
    parser.add_argument("--force", action="store_true", help="download even if an asset verifies")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    bundles = {"live", "research"} if args.bundle == "all" else {args.bundle}
    assets = [asset for asset in manifest["assets"] if asset["bundle"] in bundles]
    if not assets:
        parser.error("selected bundle has no assets")
    for asset in assets:
        download(asset, force=args.force)
    print(f"ready: {len(assets)} verified asset(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
