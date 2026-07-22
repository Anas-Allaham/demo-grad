from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import app as flask_app
import cleanvoice_service
from audio_quality import AudioQualityDecision


def test_enhance_recording_uses_pronunciation_safe_options(tmp_path, monkeypatch):
    source = tmp_path / "recording.wav"
    output = tmp_path / "recording_cleanvoice.wav"
    source.write_bytes(b"wav")
    observed = {}

    class FakeCleanvoice:
        def __init__(self, **kwargs):
            observed["init"] = kwargs

        def process(self, path, **kwargs):
            observed["path"] = path
            observed["options"] = kwargs
            Path(kwargs["output_path"]).write_bytes(b"clean wav")

    monkeypatch.setenv("CLEANVOICE_API_KEY", "test-key")
    monkeypatch.setenv("CLEANVOICE_ENABLED", "1")
    monkeypatch.setattr(cleanvoice_service, "Cleanvoice", FakeCleanvoice)

    result = cleanvoice_service.enhance_recording(source, output)

    assert result == output
    assert observed["init"]["api_key"] == "test-key"
    assert observed["path"] == str(source)
    assert observed["options"]["remove_noise"] is True
    assert observed["options"]["normalize"] is True
    assert observed["options"]["export_format"] == "wav"
    for cutting_option in (
        "fillers", "long_silences", "mouth_sounds", "breath", "stutters"
    ):
        assert observed["options"][cutting_option] is False


def test_cleanvoice_auto_enables_only_when_key_is_present(monkeypatch):
    monkeypatch.delenv("CLEANVOICE_ENABLED", raising=False)
    monkeypatch.delenv("CLEANVOICE_API_KEY", raising=False)
    assert cleanvoice_service.cleanvoice_enabled() is False
    assert cleanvoice_service.cleanvoice_configured() is False

    monkeypatch.setenv("CLEANVOICE_API_KEY", "test-key")
    assert cleanvoice_service.cleanvoice_enabled() is True
    assert cleanvoice_service.cleanvoice_configured() is True


def test_sdk_errors_are_not_exposed(tmp_path, monkeypatch):
    source = tmp_path / "recording.wav"
    source.write_bytes(b"wav")

    class FailingCleanvoice:
        def __init__(self, **_kwargs):
            pass

        def process(self, *_args, **_kwargs):
            raise RuntimeError("signed-url-with-sensitive-query")

    monkeypatch.setenv("CLEANVOICE_API_KEY", "test-key")
    monkeypatch.setattr(cleanvoice_service, "Cleanvoice", FailingCleanvoice)

    with pytest.raises(cleanvoice_service.CleanvoiceProcessingError) as caught:
        cleanvoice_service.enhance_recording(source, tmp_path / "clean.wav")

    assert "signed-url" not in str(caught.value)


def test_process_recording_uses_cleanvoice_output_before_model(tmp_path, monkeypatch):
    source = tmp_path / "recording.wav"
    source.write_bytes(b"raw wav")
    loaded_paths = []

    def fake_load(path, **_kwargs):
        loaded_paths.append(Path(path))
        return np.ones(16000, dtype=np.float32), 16000

    def fake_enhance(_source, output):
        Path(output).write_bytes(b"clean wav")
        return Path(output)

    class FakeProcessor:
        def __call__(self, *_args, **_kwargs):
            return SimpleNamespace(input_values=torch.tensor([[0.0]]))

        def batch_decode(self, _ids):
            return ["s"]

    class FakeModel:
        def __call__(self, _values):
            return SimpleNamespace(logits=torch.tensor([[[0.0, 1.0]]]))

    monkeypatch.setattr(flask_app, "convert_audio_to_wav", lambda _path: source)
    monkeypatch.setattr(flask_app.librosa, "load", fake_load)
    monkeypatch.setattr(
        flask_app,
        "analyze_audio_quality",
        lambda *_args: AudioQualityDecision(True, 1.0, [], {}),
    )
    monkeypatch.setattr(flask_app, "cleanvoice_configured", lambda: True)
    monkeypatch.setattr(flask_app, "enhance_recording", fake_enhance)
    monkeypatch.setattr(
        flask_app,
        "_apply_noise_reduction",
        lambda *_args: pytest.fail("local cleanup must not run after Cleanvoice succeeds"),
    )
    monkeypatch.setattr(flask_app, "load_wav2vec_model", lambda: None)
    monkeypatch.setattr(flask_app, "processor", FakeProcessor())
    monkeypatch.setattr(flask_app, "model", FakeModel())

    result = flask_app.process_recording(source)

    cleanvoice_path = source.with_name(source.stem + "_cleanvoice.wav")
    assert result["cleanvoice_applied"] is True
    assert result["preprocessing_pipeline"] == "cleanvoice_noise_reduction_normalization"
    assert result["reduced_audio_path"] == cleanvoice_path
    assert loaded_paths == [source, cleanvoice_path]
