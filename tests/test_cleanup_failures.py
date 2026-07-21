"""Every post-save failure path must remove all request audio artifacts."""

import io

import pytest

import app as flask_app
from audio_quality import AudioQualityDecision


@pytest.fixture
def cleanup_client(temp_db, monkeypatch):
    monkeypatch.setattr(flask_app, "RETAIN_AUDIO", False)
    return flask_app.app.test_client()


def _scorable_result(audio_path):
    # Simulate partial conversion/preprocessing artifacts. The route must know
    # how to clean these even if the result does not enumerate them.
    audio_path.with_name(audio_path.stem + "_converted.wav").write_bytes(b"converted")
    audio_path.with_name(audio_path.stem + "_reduced.wav").write_bytes(b"reduced")
    return {
        "quality_decision": AudioQualityDecision(True, 0.9, [], {}),
        "predicted_ipa": "s k u l",
        "reduced_audio_path": audio_path.with_name(audio_path.stem + "_reduced.wav"),
        "noise_reduction_applied": True,
        "preprocessing_pipeline": "test",
    }


@pytest.mark.parametrize("failure_point", ["g2p", "ffmpeg", "process", "model", "scoring", "database"])
def test_post_save_failures_leave_no_new_audio(cleanup_client, monkeypatch, failure_point):
    uploads = flask_app.UPLOAD_FOLDER
    before = set(uploads.iterdir())

    if failure_point == "g2p":
        monkeypatch.setattr(
            flask_app,
            "g2p_convert_with_metadata",
            lambda _text: (_ for _ in ()).throw(RuntimeError("injected g2p failure")),
        )
    elif failure_point == "ffmpeg":
        def fail_conversion(audio_path):
            audio_path.with_name(audio_path.stem + "_converted.wav").write_bytes(b"partial")
            raise RuntimeError("injected ffmpeg failure")

        monkeypatch.setattr(flask_app, "convert_audio_to_wav", fail_conversion)
    elif failure_point == "process":
        monkeypatch.setattr(
            flask_app,
            "process_recording",
            lambda _path: (_ for _ in ()).throw(RuntimeError("injected processing failure")),
        )
    elif failure_point == "model":
        monkeypatch.setattr(
            flask_app,
            "load_wav2vec_model",
            lambda: (_ for _ in ()).throw(RuntimeError("injected model failure")),
        )

        def fail_at_model(audio_path):
            audio_path.with_name(audio_path.stem + "_reduced.wav").write_bytes(b"partial")
            flask_app.load_wav2vec_model()

        monkeypatch.setattr(flask_app, "process_recording", fail_at_model)
    elif failure_point == "scoring":
        monkeypatch.setattr(flask_app, "process_recording", _scorable_result)
        monkeypatch.setattr(
            flask_app,
            "align_phonemes",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("injected scoring failure")),
        )
    elif failure_point == "database":
        monkeypatch.setattr(flask_app, "process_recording", _scorable_result)
        monkeypatch.setattr(
            flask_app,
            "_record_recording",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("injected database failure")),
        )

    response = cleanup_client.post(
        "/analyze",
        data={
            "text": "school",
            "user": f"cleanup-{failure_point}",
            "audio": (io.BytesIO(b"RIFFfake"), "recording.webm", "audio/webm"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 500
    assert set(uploads.iterdir()) == before


def test_missing_audio_decoder_returns_clear_400_and_cleans_upload(cleanup_client, monkeypatch):
    uploads = flask_app.UPLOAD_FOLDER
    before = set(uploads.iterdir())
    monkeypatch.setattr(flask_app, "_find_ffmpeg_executable", lambda: None)

    response = cleanup_client.post(
        "/analyze",
        data={
            "text": "school",
            "audio": (io.BytesIO(b"webm data"), "recording.webm", "audio/webm"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["code"] == "audio_decode_unavailable"
    assert "FFmpeg" in response.get_json()["error"]
    assert set(uploads.iterdir()) == before


@pytest.mark.parametrize(
    ("mimetype", "extension"),
    [
        ("audio/webm;codecs=opus", ".webm"),
        ("audio/ogg", ".ogg"),
        ("audio/wav", ".wav"),
        ("audio/mp4", ".m4a"),
    ],
)
def test_upload_extension_matches_browser_mimetype(mimetype, extension):
    assert flask_app._extension_for_mimetype(mimetype) == extension
