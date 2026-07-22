"""The release builder must include source while excluding private/runtime data."""

import importlib.util
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _load_packager():
    spec = importlib.util.spec_from_file_location(
        "package_release", ROOT / "scripts" / "package_release.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_packager_exclusion_policy_covers_private_artifacts():
    package = _load_packager()
    excluded = [
        ".env", ".env.production", "credentials.json", "app.db",
        "data/history.sqlite", "uploads/recording.webm", "model/model.safetensors",
        "model/training_args.bin", ".pytest_cache/state", "CODEX_TASK.md",
        "unexpected-local-file.txt",
    ]
    for relative in excluded:
        assert package._excluded(package.BASE_DIR / relative), relative

    included = [
        ".env.example", "app.py", "cleanvoice_service.py", "data/seed_sentences.txt",
        "tests/fixtures/speech_sample.wav", "uploads/.gitkeep",
    ]
    for relative in included:
        assert not package._excluded(package.BASE_DIR / relative), relative


def test_packager_builds_and_validates_clean_archive(tmp_path, monkeypatch):
    package = _load_packager()
    output = tmp_path / "deliverable.zip"
    monkeypatch.setattr(package, "MANIFEST_PATH", tmp_path / "MANIFEST.txt")

    hashed_count = package.write_archive(output)
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())

    assert "MANIFEST.txt" in names
    assert "app.py" in names
    assert "cleanvoice_service.py" in names
    assert "tests/test_packaging.py" in names
    assert len(names) == hashed_count + 1
    assert ".env" not in names
    assert "app.db" not in names
    assert "model/my_wav2vec2_phoneme_model/model.safetensors" not in names
    assert "model/my_wav2vec2_phoneme_model/training_args.bin" not in names
    assert not any(name.startswith("uploads/") and name != "uploads/.gitkeep" for name in names)
