"""
Inference module — calls the remote TRIBE v2 backend (Colab/Modal).
Falls back to a mock response when no backend URL is set.
"""
import os, io, base64, json, requests
import numpy as np
from typing import Union

BACKEND_URL = os.getenv("TRIBE_BACKEND_URL", "")  # Set this to your ngrok/Modal URL

# Ordered list of all ROI keys we return (must match Colab server output)
ROI_KEYS = [
    "V1", "V2_V3", "FFA", "PPA", "EBA", "LOC", "MT_V5",
    "A1", "Belt_Areas",
    "Broca", "Wernicke", "VWFA",
    "TPJ", "Amygdala", "mPFC",
    "Hippocampus", "RSC"
]


def _mock_response(modality: str, hint: str = "") -> dict:
    """Deterministic mock so the UI still works without a live backend."""
    np.random.seed(abs(hash(hint)) % (2**31))
    activations = {}
    # Bias activations by modality
    boosts = {
        "image": ["V1", "V2_V3", "FFA", "PPA", "LOC", "EBA"],
        "video": ["MT_V5", "V1", "V2_V3", "A1", "Belt_Areas", "TPJ"],
        "text":  ["Broca", "Wernicke", "VWFA", "mPFC", "Hippocampus"],
        "audio": ["A1", "Belt_Areas", "Broca", "Wernicke"],
    }.get(modality, [])
    for roi in ROI_KEYS:
        base = float(np.random.beta(2, 5))
        if roi in boosts:
            base = min(1.0, base + float(np.random.uniform(0.3, 0.5)))
        activations[roi] = round(base, 4)
    return {"activations": activations, "modality": modality, "source": "mock"}


def _encode(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _call_backend(payload: dict, timeout: int = 60) -> dict:
    resp = requests.post(
        f"{BACKEND_URL.rstrip('/')}/predict",
        json=payload,
        timeout=timeout
    )
    resp.raise_for_status()
    return resp.json()


# ── Public API ────────────────────────────────────────────────────────────────

def predict_from_image(image_bytes: bytes) -> dict:
    if not BACKEND_URL:
        return _mock_response("image", hint=str(image_bytes[:64]))
    return _call_backend({"modality": "image", "data": _encode(image_bytes)})


def predict_from_video(video_bytes: bytes) -> dict:
    if not BACKEND_URL:
        return _mock_response("video", hint=str(video_bytes[:64]))
    return _call_backend({"modality": "video", "data": _encode(video_bytes)}, timeout=120)


def predict_from_text(text: str) -> dict:
    if not BACKEND_URL:
        return _mock_response("text", hint=text[:80])
    return _call_backend({"modality": "text", "text": text})
