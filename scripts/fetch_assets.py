#!/usr/bin/env python3
"""Download repo assets declared in ImageAI-master/data-assets.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "ImageAI-master" / "data-assets.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target.with_suffix(target.suffix + ".tmp")
    with urllib.request.urlopen(url) as response, temp_target.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    temp_target.replace(target)


def extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                target = (destination / member).resolve()
                target.relative_to(destination.resolve())
            zf.extractall(destination)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                target = (destination / member.name).resolve()
                target.relative_to(destination.resolve())
            tf.extractall(destination)
        return
    raise ValueError(f"unsupported archive format: {archive}")


def resolve_repo_path(raw_path: str) -> Path:
    path = (REPO_ROOT / raw_path).resolve()
    try:
        path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"path escapes repository root: {raw_path}") from exc
    return path


def fetch_asset(asset: dict[str, str], force: bool) -> None:
    path = resolve_repo_path(asset["path"])
    url = asset["url"]
    expected_sha = asset.get("sha256", "").strip().lower()

    if not url or "example.com" in url:
        print(f"SKIP {asset['path']} - set a real URL in the manifest")
        return

    if path.exists() and not force:
        if expected_sha and sha256_file(path) != expected_sha:
            print(f"REFETCH {asset['path']} - checksum mismatch")
        else:
            print(f"OK {asset['path']}")
            return

    print(f"GET {asset['path']}")
    download(url, path)

    if expected_sha:
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            path.unlink(missing_ok=True)
            raise ValueError(
                f"checksum mismatch for {asset['path']}: "
                f"expected {expected_sha}, got {actual_sha}"
            )

    extract_to = asset.get("extract")
    if extract_to:
        destination = resolve_repo_path(extract_to)
        print(f"EXTRACT {asset['path']} -> {extract_to}")
        extract_archive(path, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        type=Path,
        help="Path to the JSON asset manifest.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download assets even when the local file already exists.",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    for asset in manifest.get("assets", []):
        fetch_asset(asset, force=args.force)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
