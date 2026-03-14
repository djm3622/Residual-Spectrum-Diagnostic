#!/usr/bin/env python3
"""Download original FNO Navier-Stokes full-trajectory MAT datasets."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile, is_zipfile

# Source discussed in the original FNO repo dataset folder.
# Default file contains full trajectories used by many follow-up studies.
DATASETS = {
    "v1e-3_n5000_t50": {
        "filename": "NavierStokes_V1e-3_N5000_T50.mat",
        "archive_name": "NavierStokes_V1e-3_N5000_T50.zip",
        "drive_id": "1r3idxpsHa21ijhlu3QQ1hVuXcqnBTO7d",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="download_fno_ns_mat.py",
        description=(
            "Download original FNO Navier-Stokes trajectory MAT data from Google Drive. "
            "Default target is NavierStokes_V1e-3_N5000_T50.mat."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS.keys()),
        default="v1e-3_n5000_t50",
        help="Named dataset preset to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="external_data/fno",
        help="Destination directory for archive and extracted MAT file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Delete local archive/MAT first and re-download from scratch.",
    )
    return parser.parse_args()


def _human_size(num_bytes: int) -> str:
    size = float(max(0, int(num_bytes)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def _run_command(cmd: list[str]) -> bool:
    print("[download]", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    return completed.returncode == 0


def _validate_archive(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    return is_zipfile(path)


def _download_with_gdown(drive_id: str, archive_path: Path) -> bool:
    view_url = f"https://drive.google.com/file/d/{drive_id}/view"
    candidates: list[list[str]] = []
    gdown_bin = shutil.which("gdown")
    if gdown_bin:
        candidates.append([gdown_bin, "--fuzzy", "--continue", view_url, "-O", str(archive_path)])
    candidates.append([sys.executable, "-m", "gdown", "--fuzzy", "--continue", view_url, "-O", str(archive_path)])

    for cmd in candidates:
        if not _run_command(cmd):
            continue
        if _validate_archive(archive_path):
            return True
        print("[warn] gdown command finished but archive is not a valid ZIP yet.")
    return False


def _download_with_curl(drive_id: str, archive_path: Path) -> bool:
    curl_bin = shutil.which("curl")
    if not curl_bin:
        return False

    # Public-file fallback URL. gdown remains the preferred route.
    url = f"https://drive.google.com/uc?export=download&id={drive_id}"
    cmd = [
        curl_bin,
        "-fL",
        "--retry",
        "5",
        "--retry-delay",
        "2",
        "--continue-at",
        "-",
        "--output",
        str(archive_path),
        url,
    ]
    if not _run_command(cmd):
        return False
    return _validate_archive(archive_path)


def _ensure_archive(meta: dict[str, str], archive_path: Path) -> None:
    if _validate_archive(archive_path):
        print(
            f"[ok] using existing archive: {archive_path} "
            f"({_human_size(archive_path.stat().st_size)})"
        )
        return

    drive_id = str(meta["drive_id"])
    print("[info] attempting Google Drive download with gdown (resumable).")
    if _download_with_gdown(drive_id, archive_path):
        print(
            f"[ok] archive ready via gdown: {archive_path} "
            f"({_human_size(archive_path.stat().st_size)})"
        )
        return

    print("[warn] gdown download unavailable/failed; trying curl fallback.")
    if _download_with_curl(drive_id, archive_path):
        print(
            f"[ok] archive ready via curl fallback: {archive_path} "
            f"({_human_size(archive_path.stat().st_size)})"
        )
        return

    raise RuntimeError(
        "Download failed. Install gdown for robust Google Drive support "
        "(python3 -m pip install gdown) and retry."
    )


def _extract_expected_mat(archive_path: Path, out_dir: Path, expected_name: str) -> Path:
    expected_path = out_dir / expected_name
    print(f"[extract] {archive_path} -> {out_dir}")
    with ZipFile(archive_path, "r") as zf:
        members = zf.namelist()
        mat_members = [name for name in members if name.lower().endswith(".mat")]
        if not mat_members:
            raise RuntimeError(f"Archive contains no .mat file: {archive_path}")

        member = ""
        for candidate in mat_members:
            if Path(candidate).name == expected_name:
                member = candidate
                break
        if not member:
            member = mat_members[0]
            print(f"[warn] expected MAT name not found; using: {member}")

        zf.extract(member, path=out_dir)

    extracted_path = out_dir / member
    if extracted_path != expected_path:
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        if expected_path.exists():
            expected_path.unlink()
        shutil.move(str(extracted_path), str(expected_path))

        # Remove empty extraction directories when archive had nested folders.
        parent = extracted_path.parent
        while parent != out_dir and parent.exists():
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    return expected_path


def _verify_mat_readable(mat_path: Path, dataset_key: str = "u") -> tuple[bool, str]:
    if not mat_path.exists() or not mat_path.is_file():
        return False, f"MAT file not found: {mat_path}"

    scipy_error = ""
    try:
        from scipy.io import whosmat

        entries = whosmat(mat_path)
        names = {name: tuple(int(dim) for dim in shape) for name, shape, _ in entries}
        if dataset_key in names:
            shape = names[dataset_key]
            if len(shape) == 4:
                return True, f"scipy header key '{dataset_key}' shape={shape}"
            return False, f"key '{dataset_key}' has non-4D shape {shape}"
    except Exception as exc:  # noqa: BLE001 - verification fallback path
        scipy_error = str(exc)

    h5_error = ""
    try:
        import h5py

        with h5py.File(mat_path, "r") as handle:
            if dataset_key not in handle:
                keys = sorted(handle.keys())
                return False, f"h5py keys missing '{dataset_key}'; found keys={keys}"
            dataset = handle[dataset_key]
            shape = tuple(int(dim) for dim in dataset.shape)
            if len(shape) != 4:
                return False, f"h5py key '{dataset_key}' has non-4D shape {shape}"
            return True, f"h5py key '{dataset_key}' shape={shape}"
    except Exception as exc:  # noqa: BLE001 - final verification error capture
        h5_error = str(exc)

    detail = "; ".join(part for part in (scipy_error, h5_error) if part)
    return False, detail or "unrecognized MAT format"


def main() -> None:
    args = parse_args()
    meta = DATASETS[args.dataset]

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mat_path = out_dir / str(meta["filename"])
    archive_path = out_dir / str(meta["archive_name"])

    print(f"[info] output directory: {out_dir}")
    print(f"[info] target MAT file: {mat_path.name}")

    if args.force:
        if mat_path.exists():
            print(f"[force] removing {mat_path}")
            mat_path.unlink()
        if archive_path.exists():
            print(f"[force] removing {archive_path}")
            archive_path.unlink()

    if mat_path.exists():
        ok, reason = _verify_mat_readable(mat_path)
        if ok:
            print(f"[ok] existing MAT is readable ({reason}).")
            print(f"[ok] final MAT path: {mat_path}")
            return
        print(f"[warn] existing MAT failed readability check: {reason}")
        mat_path.unlink()

    _ensure_archive(meta, archive_path)

    if not _validate_archive(archive_path):
        raise RuntimeError(
            f"Archive is missing or invalid after download: {archive_path}. "
            "Retry with --force."
        )

    extracted_mat = _extract_expected_mat(archive_path, out_dir, str(meta["filename"]))
    ok, reason = _verify_mat_readable(extracted_mat)
    if not ok:
        raise RuntimeError(
            "Downloaded MAT file failed readability check: "
            f"{reason}. File: {extracted_mat}"
        )

    print(f"[ok] MAT verified ({reason}).")
    print(
        f"[ok] final MAT path: {extracted_mat} "
        f"({_human_size(extracted_mat.stat().st_size)})"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
