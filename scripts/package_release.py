"""Build and validate the corrected source archive reproducibly.

Text files are normalized to LF in memory. MANIFEST.txt hashes those exact
archive bytes, and validation reopens the ZIP before reporting success.
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = BASE_DIR.parent / "pronunciation_web_app_integrated_corrected.zip"
MANIFEST_PATH = BASE_DIR / "MANIFEST.txt"

TEXT_SUFFIXES = {
    ".css", ".env", ".html", ".ini", ".js", ".json", ".md", ".py", ".txt",
}
TEXT_NAMES = {".gitignore", ".env.example"}
CACHE_DIRS = {".git", ".pytest_cache", ".mypy_cache", ".ruff_cache", "__pycache__"}
LOCAL_TOOL_DIRS = {".claude", ".sixth", ".vscode"}
ALLOWED_ROOT_DIRS = {
    "data", "g2p_pipeline_split_v2", "model", "scripts", "static", "templates",
    "tests", "uploads", "voice-filtering",
}
ALLOWED_ROOT_FILES = {
    ".env.example", ".gitignore", "README.md", "TEST_OUTPUT.txt", "app.py",
    "app_datetime.py", "assessment.py", "audio_quality.py", "cleanvoice_service.py",
    "content.py", "db.py",
    "g2p_service.py", "mastery.py", "phoneme_vectors.py",
    "phoneme_vectors_professional.py", "pytest.ini", "requirements-minimal.txt",
    "requirements.txt", "scoring.py", "services.py", "test_g2p.py", "tokenization.py",
}
MODEL_WEIGHT_NAMES = {
    "model.safetensors", "pytorch_model.bin", "tf_model.h5", "flax_model.msgpack",
}
RECORDING_SUFFIXES = {".webm", ".ogg", ".m4a", ".opus", ".mp3", ".flac", ".wav"}
SECRET_NAMES = {"credentials.json", "secrets.json", "service-account.json"}
SECRET_SUFFIXES = {".key", ".pem", ".p12", ".pfx"}
DATABASE_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


def _excluded(path: Path) -> bool:
    relative = path.relative_to(BASE_DIR)
    parts = relative.parts
    posix = relative.as_posix()
    name = path.name
    lower_name = name.lower()

    if parts[0] not in ALLOWED_ROOT_DIRS and name not in ALLOWED_ROOT_FILES:
        return True
    if any(part in CACHE_DIRS | LOCAL_TOOL_DIRS for part in parts):
        return True
    if name in {"MANIFEST.txt", "CODEX_TASK.md"} or path.suffix.lower() == ".zip":
        return True
    if (lower_name.startswith(".env") and lower_name != ".env.example") or path.suffix.lower() == ".env":
        return True
    if lower_name in SECRET_NAMES or path.suffix.lower() in SECRET_SUFFIXES:
        return True
    if path.suffix.lower() in DATABASE_SUFFIXES or lower_name.endswith((".db-wal", ".db-shm", ".db-bak")):
        return True
    if (
        name in MODEL_WEIGHT_NAMES
        or path.suffix.lower() in {".safetensors", ".ckpt"}
        or (parts[0] == "model" and path.suffix.lower() == ".bin")
    ):
        return True
    if posix.startswith("uploads/") and posix != "uploads/.gitkeep":
        return True
    if path.suffix.lower() in RECORDING_SUFFIXES and not posix.startswith("tests/fixtures/"):
        return True
    if name.endswith("_converted.wav") or name.endswith("_reduced.wav"):
        return True
    if path.suffix.lower() in {".pyc", ".pyo"}:
        return True
    return False


def _archive_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    if path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_NAMES:
        text = raw.decode("utf-8")
        return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return raw


def collect_members() -> Dict[str, bytes]:
    members: Dict[str, bytes] = {}
    for path in sorted(BASE_DIR.rglob("*")):
        if not path.is_file() or _excluded(path):
            continue
        members[path.relative_to(BASE_DIR).as_posix()] = _archive_bytes(path)
    return members


def build_manifest(members: Dict[str, bytes]) -> bytes:
    lines = [
        "# File manifest for pronunciation_web_app_integrated_corrected.zip",
        "# Hashes and byte counts are from the exact normalized archive members.",
        "",
        "sha256                                                           bytes  path",
    ]
    for name, data in sorted(members.items()):
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {len(data):10d}  {name}")
    lines.extend(["", f"Total files hashed: {len(members)}", ""])
    return "\n".join(lines).encode("utf-8")


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(2026, 7, 18, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def write_archive(output: Path) -> int:
    members = collect_members()
    manifest = build_manifest(members)
    MANIFEST_PATH.write_bytes(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(members.items()):
            archive.writestr(_zip_info(name), data)
        archive.writestr(_zip_info("MANIFEST.txt"), manifest)
    validate_archive(output)
    return len(members)


def _parse_manifest(data: bytes) -> Dict[str, tuple[str, int]]:
    expected: Dict[str, tuple[str, int]] = {}
    for line in data.decode("utf-8").splitlines():
        if not line or line.startswith("#") or line.startswith("sha256") or line.startswith("Total"):
            continue
        digest, size, name = line.split(maxsplit=2)
        expected[name] = (digest, int(size))
    return expected


def validate_archive(output: Path) -> None:
    with zipfile.ZipFile(output, "r") as archive:
        names = set(archive.namelist())
        if "MANIFEST.txt" not in names:
            raise RuntimeError("Archive has no MANIFEST.txt")
        expected = _parse_manifest(archive.read("MANIFEST.txt"))
        actual_names = names - {"MANIFEST.txt"}
        if set(expected) != actual_names:
            missing = sorted(actual_names - set(expected))
            extra = sorted(set(expected) - actual_names)
            raise RuntimeError(f"Manifest member mismatch: missing={missing}, extra={extra}")

        for name, (digest, size) in expected.items():
            data = archive.read(name)
            if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError(f"Manifest validation failed for {name}")
            suffix = Path(name).suffix.lower()
            if suffix in TEXT_SUFFIXES or Path(name).name in TEXT_NAMES:
                if b"\r" in data:
                    raise RuntimeError(f"Text line endings were not normalized: {name}")

        forbidden = [
            name for name in names
            if (
                Path(name).parts[0] not in ALLOWED_ROOT_DIRS
                and Path(name).name not in ALLOWED_ROOT_FILES | {"MANIFEST.txt"}
            )
            or (
                Path(name).name.lower().startswith(".env")
                and Path(name).name.lower() != ".env.example"
            )
            or Path(name).name.lower() in SECRET_NAMES
            or Path(name).suffix.lower() in SECRET_SUFFIXES
            or Path(name).suffix.lower() in DATABASE_SUFFIXES
            or Path(name).name.lower().endswith((".db-wal", ".db-shm", ".db-bak"))
            or Path(name).name in MODEL_WEIGHT_NAMES
            or Path(name).suffix.lower() in {".safetensors", ".ckpt"}
            or (Path(name).parts[0] == "model" and Path(name).suffix.lower() == ".bin")
            or (name.startswith("uploads/") and name != "uploads/.gitkeep")
            or (
                Path(name).suffix.lower() in RECORDING_SUFFIXES
                and not name.startswith("tests/fixtures/")
            )
            or any(part in CACHE_DIRS | LOCAL_TOOL_DIRS for part in Path(name).parts)
        ]
        if forbidden:
            raise RuntimeError(f"Forbidden archive members: {forbidden}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    count = write_archive(output)
    print(f"Archive: {output}")
    print(f"Manifest validation: OK ({count} hashed members; all hashes and sizes match)")
    print("Exclusions: OK (no secrets, user DB, uploaded recordings, model weights, or caches)")


if __name__ == "__main__":
    main()
