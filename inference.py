"""
Inference module — calls the remote TRIBE v2 backend (Colab/Modal).
Falls back to a mock response when no backend URL is set or on error.
"""
import os, io, base64, json, logging, requests
import numpy as np
from typing import Union

log = logging.getLogger(__name__)

def _get_backend_url() -> str:
    """Read dynamically so sidebar changes take effect immediately."""
    return os.getenv("TRIBE_BACKEND_URL", "").strip()

# Ordered list of all ROI keys we return (must match Colab server output)
ROI_KEYS = [
    "V1", "V2_V3", "FFA", "PPA", "EBA", "LOC", "MT_V5",
    "A1", "Belt_Areas",
    "Broca", "Wernicke", "VWFA",
    "TPJ", "Amygdala", "mPFC",
    "Hippocampus", "RSC",
    "NAcc", "OFC", "Insula", "ACC", "VTA", "Hypothalamus"
]


def _mock_response(modality: str, hint: str = "", fallback: bool = False) -> dict:
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
    source = "mock_fallback" if fallback else "mock"
    return {"activations": activations, "modality": modality, "source": source}


def _encode(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _call_backend(payload: dict, timeout: int = 60) -> dict:
    url = _get_backend_url()
    resp = requests.post(
        f"{url.rstrip('/')}/predict",
        json=payload,
        timeout=timeout
    )
    resp.raise_for_status()
    return resp.json()


# ── Public API ────────────────────────────────────────────────────────────────

def predict_from_image(image_bytes: bytes) -> dict:
    url = _get_backend_url()
    if not url:
        return _mock_response("image", hint=str(image_bytes[:64]))
    try:
        return _call_backend({"modality": "image", "data": _encode(image_bytes)})
    except Exception as e:
        log.warning("Backend error for image, falling back to demo: %s", e)
        result = _mock_response("image", hint=str(image_bytes[:64]), fallback=True)
        result["backend_error"] = str(e)
        return result


def predict_from_video(video_bytes: bytes) -> dict:
    url = _get_backend_url()
    if not url:
        return _mock_response("video", hint=str(video_bytes[:64]))
    try:
        return _call_backend({"modality": "video", "data": _encode(video_bytes)}, timeout=120)
    except Exception as e:
        log.warning("Backend error for video, falling back to demo: %s", e)
        result = _mock_response("video", hint=str(video_bytes[:64]), fallback=True)
        result["backend_error"] = str(e)
        return result


def predict_from_text(text: str) -> dict:
    url = _get_backend_url()
    if not url:
        return _mock_response("text", hint=text[:80])
    try:
        return _call_backend({"modality": "text", "text": text})
    except Exception as e:
        log.warning("Backend error for text, falling back to demo: %s", e)
        result = _mock_response("text", hint=text[:80], fallback=True)
        result["backend_error"] = str(e)
        return result
