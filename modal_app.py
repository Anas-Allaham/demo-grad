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

# Predictive OOV G2P (ByT5). Weights are baked into a cached image layer at
# build time so the running container never reaches the network. Keep this in
# sync with oov_g2p.py DEFAULT_MODEL.
HF_CACHE_DIR = "/opt/hf-cache"
OOV_G2P_MODEL = "charsiu/g2p_multilingual_byT5_tiny_16_layers_100"


def _bake_oov_g2p_model() -> None:
    """Download the ByT5 OOV G2P weights into the image's HF cache at build time."""
    import os as _os

    _os.environ["HF_HOME"] = HF_CACHE_DIR
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    T5ForConditionalGeneration.from_pretrained(OOV_G2P_MODEL)
    # ByT5's byte tokenizer needs no vocab files; resolving it here just warms
    # the config so the runtime (offline) load never touches the network.
    try:
        AutoTokenizer.from_pretrained(OOV_G2P_MODEL)
    except Exception:
        from transformers import ByT5Tokenizer

        ByT5Tokenizer()

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
            # Predictive OOV G2P: enabled, with weights served from the baked
            # HF cache below.
            "OOV_G2P_ENABLED": "1",
            "OOV_G2P_MODEL": OOV_G2P_MODEL,
            "HF_HOME": HF_CACHE_DIR,
            # The Xet transfer backend can stall on unauthenticated pulls; use
            # the standard resolver for the deterministic build-time download.
            "HF_HUB_DISABLE_XET": "1",
        }
    )
    # Bake the ByT5 weights into their own cached layer (needs network at build
    # time only), then pin the runtime offline so a request never fetches.
    .run_function(_bake_oov_g2p_model)
    .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
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
