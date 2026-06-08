#!/usr/bin/env python3
"""Download MiMo-V2-Flash HF weights to GCS, one file at a time.

Each file is downloaded to /tmp (max 4.3 GB), uploaded to GCS, then deleted.
Keeps peak disk usage under 10 GB at any time.

Usage:
    python3 scripts/download_mimo_v2_flash_to_gcs.py
    # or to resume:
    python3 scripts/download_mimo_v2_flash_to_gcs.py --skip-existing

GCS output: gs://jingnw-mimo-v2-5-pro-us-central1/mimo-v2-flash-hf-weights/
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfFileSystem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO = "XiaomiMiMo/MiMo-V2-Flash"
GCS_DEST = "gs://jingnw-mimo-v2-5-pro-us-central1/mimo-v2-flash-hf-weights"
TMP_DIR = Path("/tmp/mimo-v2-flash-dl")
MAX_WORKERS = 4  # parallel downloads (each file up to 4.3 GB)


def gcs_exists(gcs_path: str) -> bool:
    result = subprocess.run(
        ["gsutil", "-q", "stat", gcs_path],
        capture_output=True,
    )
    return result.returncode == 0


def download_one(hf_path: str, filename: str, skip_existing: bool) -> tuple[str, str, float]:
    """Download one file from HF → /tmp → GCS. Returns (filename, status, elapsed)."""
    gcs_path = f"{GCS_DEST}/{filename}"
    local_path = TMP_DIR / filename

    if skip_existing and gcs_exists(gcs_path):
        return filename, "skipped", 0.0

    t0 = time.time()
    try:
        # Step 1: download from HF to /tmp
        log.info("DL  %s", filename)
        fs = HfFileSystem()
        with fs.open(f"{REPO}/{filename}", "rb") as src, open(local_path, "wb") as dst:
            while chunk := src.read(8 * 1024 * 1024):
                dst.write(chunk)

        # Step 2: upload to GCS
        log.info("GCS %s  (%.1f GB)", filename, local_path.stat().st_size / 1e9)
        result = subprocess.run(
            ["gsutil", "-q", "cp", str(local_path), gcs_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"gsutil cp failed: {result.stderr}")

        elapsed = time.time() - t0
        return filename, "ok", elapsed

    except Exception as e:
        return filename, f"ERROR: {e}", time.time() - t0

    finally:
        if local_path.exists():
            local_path.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files already present in GCS")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"Parallel downloads (default {MAX_WORKERS})")
    args = parser.parse_args()

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # List all files in HF repo
    log.info("Listing files in %s ...", REPO)
    fs = HfFileSystem()
    all_files = fs.ls(REPO, detail=True)

    # Separate safetensors from metadata files
    safetensors = sorted(f["name"].split("/")[-1] for f in all_files
                         if f["name"].endswith(".safetensors"))
    metadata = sorted(f["name"].split("/")[-1] for f in all_files
                      if not f["name"].endswith(".safetensors")
                      and not f["name"].split("/")[-1].startswith("."))

    total_size = sum(f.get("size", 0) for f in all_files) / 1e9
    log.info("Found %d safetensors + %d metadata files, %.1f GB total",
             len(safetensors), len(metadata), total_size)

    # Upload metadata files first (tiny, serial)
    log.info("Uploading %d metadata files ...", len(metadata))
    for fname in metadata:
        gcs_path = f"{GCS_DEST}/{fname}"
        if args.skip_existing and gcs_exists(gcs_path):
            log.info("  skip %s (exists)", fname)
            continue
        local = TMP_DIR / fname
        with fs.open(f"{REPO}/{fname}", "rb") as src, open(local, "wb") as dst:
            dst.write(src.read())
        subprocess.run(["gsutil", "-q", "cp", str(local), gcs_path], check=True)
        local.unlink()
        log.info("  done %s", fname)

    # Download safetensors in parallel
    log.info("Downloading %d safetensors with %d workers ...", len(safetensors), args.workers)
    done = 0
    errors = []
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_one, f"{REPO}/{fname}", fname, args.skip_existing): fname
            for fname in safetensors
        }
        for future in as_completed(futures):
            fname, status, elapsed = future.result()
            done += 1
            if status.startswith("ERROR"):
                errors.append((fname, status))
                log.error("[%d/%d] FAIL %s — %s", done, len(safetensors), fname, status)
            else:
                log.info("[%d/%d] %-8s %s  (%.0fs)", done, len(safetensors), status, fname, elapsed)

    elapsed_total = time.time() - t_start
    log.info("Done: %d/%d OK, %d errors, %.1f min total",
             done - len(errors), len(safetensors), len(errors), elapsed_total / 60)

    if errors:
        log.error("Failed files:")
        for fname, err in errors:
            log.error("  %s: %s", fname, err)
        sys.exit(1)

    log.info("All files at: %s", GCS_DEST)


if __name__ == "__main__":
    main()
