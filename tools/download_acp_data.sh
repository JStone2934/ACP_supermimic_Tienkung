#!/usr/bin/env bash
# ACP-G1 assets only; run in the conda environment. No Docker or Isaac Lab needed.
set -euo pipefail
cd "$(dirname "$0")/.."

PHP_REV=c7c9ce3fc75a8754a7e403c58de3b6933254d01b
CACHE=.cache/setup/php
mkdir -p "$CACHE/scripts"
for file in release-assets.json scripts/download_assets.py; do
  curl -fsSL --retry 4 --retry-all-errors --connect-timeout 20 --max-time 120 \
    "https://raw.githubusercontent.com/amazon-far/php_parkour/$PHP_REV/$file" \
    -o "$CACHE/$file.part"
  mv "$CACHE/$file.part" "$CACHE/$file"
done
# The upstream downloader checks both archive and extracted file SHA-256 hashes.
for component in motions-locomotion motions-low-step motions-high-step motions-low-climb-76 motions-high-climb-76; do
  python "$CACHE/scripts/download_assets.py" "$component" \
    --destination "data/php_parkour/$component"
done

python - <<'PY'
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

revision = "ce1572906efe6157840e8474d5a0d7aa87481e74"
dataset = "lvhaidong/LAFAN1_Retargeting_Dataset"
root = Path("data/g1_description")
root.mkdir(parents=True, exist_ok=True)

def curl(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    subprocess.run(["curl", "-fsSL", "--retry", "4", "--retry-all-errors",
                    "--connect-timeout", "20", "--max-time", "180", url, "-o", str(part)], check=True)
    part.replace(target)

entries = []
for folder in ("robot_description/g1", "robot_description/g1/meshes"):
    listing = Path(".cache/setup") / (folder.replace("/", "_") + ".json")
    curl(f"https://huggingface.co/api/datasets/{dataset}/tree/{revision}/{folder}", listing)
    entries.extend(x for x in json.loads(listing.read_text()) if x["type"] == "file"
                   and (x["path"].endswith(".STL") or x["path"].endswith("/g1_29dof_rev_1_0.urdf")))

def matches(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    data = path.read_bytes()
    if "lfs" in entry:
        return hashlib.sha256(data).hexdigest() == entry["lfs"]["oid"]
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() == entry["oid"]

def fetch(entry):
    path = root / Path(entry["path"]).relative_to("robot_description/g1")
    if not matches(path, entry):
        curl(f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{entry['path']}", path)
    if not matches(path, entry):
        raise RuntimeError(f"Checksum mismatch: {path}")

with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(fetch, entries))
(root / "download-manifest.json").write_text(json.dumps({"revision": revision, "files": entries}, indent=2))
print(f"Verified {len(entries)} G1 description files")
PY

python tools/urdf_to_mjcf.py
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" \
  python -m supermimic.acp.make_labels --workers "${LABEL_WORKERS:-16}"
