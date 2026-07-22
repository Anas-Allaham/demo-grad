"""Modal deployment entry point for the Flask pronunciation application.

Development:
    modal serve modal_app.py

Production:
    modal deploy modal_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal


APP_NAME = "pronunciation-coach"
APP_DIR = "/root/pronunciation-app"
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model" / "my_wav2vec2_phoneme_model"
MODEL_REMOTE_DIR = f"{APP_DIR}/model/my_wav2vec2_phoneme_model"

# Never upload local secrets, recordings, databases, caches, or development
# artifacts. The model is copied in its own cached layer below, so source-only
# changes do not require uploading the 378 MB weight again.
IMAGE_IGNORES = [
    ".git",
    ".git/**",
    ".env",
    ".env.*",
    "!.env.example",
    ".pytest_cache",
    ".pytest_cache/**",
    "**/__pycache__",
    "**/__pycache__/**",
    "**/*.pyc",
    "tests",
    "tests/**",
    "app.db",
    "app.db-*",
    "app.db.*",
    "uploads/**",
    "model/**",
    "TEST_OUTPUT.txt",
    "CODEX_TASK.md",
]

runtime_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "torch==2.11.0",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install_from_requirements(str(PROJECT_DIR / "requirements-deploy.txt"))
    .env(
        {
            "APP_DB_PATH": "/data/app.db",
            "CLEANVOICE_ENABLED": "1",
            "CLEANVOICE_STRICT": "0",
            "RETAIN_AUDIO": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir(
        str(MODEL_DIR),
        remote_path=MODEL_REMOTE_DIR,
        copy=True,
    )
    .add_local_dir(
        str(PROJECT_DIR),
        remote_path=APP_DIR,
        copy=True,
        ignore=IMAGE_IGNORES,
    )
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(
    "pronunciation-data",
    create_if_missing=True,
)


@app.function(
    image=runtime_image,
    secrets=[modal.Secret.from_name("pronunciation-secrets")],
    volumes={"/data": data_volume},
    cpu=1.0,
    memory=4096,
    timeout=600,
    scaledown_window=60,
    max_containers=1,
)
@modal.wsgi_app()
def web():
    """Expose the existing Flask instance at a public Modal HTTPS URL."""
    os.chdir(APP_DIR)
    if APP_DIR not in sys.path:
        sys.path.insert(0, APP_DIR)

    from app import app as flask_app
    import db

    # A new persistent Volume starts with an empty schema. Populate only the
    # public exercise bank on first boot; never copy local users or attempts.
    if db.count_exercise_bank() == 0:
        from scripts.build_exercise_bank import main as build_exercise_bank

        build_exercise_bank()

    return flask_app
