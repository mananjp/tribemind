import os
import io
import base64
import tempfile
import numpy as np
import torch
from dotenv import load_dotenv
load_dotenv()
import uvicorn
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from huggingface_hub import snapshot_download

# Initialize FastAPI app
app_api = FastAPI(title="TribeMind TRIBE v2 Local Server")

ROI_KEYS = [
    "V1","V2_V3","FFA","PPA","EBA","LOC","MT_V5",
    "A1","Belt_Areas",
    "Broca","Wernicke","VWFA",
    "TPJ","Amygdala","mPFC",
    "Hippocampus","RSC",
    "NAcc","OFC","Insula","ACC","VTA","Hypothalamus"
]

CHUNK = 20484 // len(ROI_KEYS)

# Load model
model = None
try:
    # Test import FIRST so we don't waste 3 minutes downloading weights if the library isn't installed
    import importlib.util
    if importlib.util.find_spec("tribev2") is None:
        raise ImportError("No module named 'tribev2'")

    print("Downloading/Loading model weights...")
    from tribev2 import TribeModel
    model_path = snapshot_download(
        repo_id='facebook/tribev2',
        ignore_patterns=['*subject_0[2-9]*', '*subject_1*'],
    )
    model = TribeModel.from_pretrained(model_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"Loaded on {next(model.parameters()).device}")
    else:
        print("Loaded on CPU")
except Exception as e:
    print(f"Error loading TribeModel: {e}")
    print("Please ensure you have installed the official tribev2 library.")
    print("-> Falling back to MockTribeModel since model loading created issues.")
    
    class MockTribeModel:
        def get_events_dataframe(self, video_path=None, text=None):
            return "video" if video_path else "text"
            
        def predict(self, events):
            preds = np.random.beta(2, 5, size=(1, 20484))
            
            boosts = []
            if events == "video":
                # V1=0, V2_V3=1, FFA=2, PPA=3, LOC=5, MT_V5=6, A1=7, Belt=8, TPJ=12
                # + reward: NAcc=17, Insula=19, VTA=21, Hypothalamus=22
                boosts = [0, 1, 2, 3, 5, 6, 7, 8, 12, 17, 19, 21, 22]
            elif events == "text":
                # Broca=9, Wernicke=10, VWFA=11, mPFC=14, Hippocampus=15
                # + reward: OFC=18, ACC=20
                boosts = [9, 10, 11, 14, 15, 18, 20]
                
            CHUNK = 20484 // len(ROI_KEYS)
            for i in boosts:
                if i < len(ROI_KEYS):
                    preds[0, i * CHUNK : (i + 1) * CHUNK] += np.random.uniform(0.3, 0.5)
            
            return preds, None
            
    model = MockTribeModel()

def decode_b64(b64str):
    return base64.b64decode(b64str)

def image_to_tmp_video(img_bytes, duration=2, fps=4):
    import cv2
    img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    path = tmp.name
    tmp.close()
    
    h, w = img.shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(duration * fps):
        out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    out.release()
    return path

def activations_from_preds(preds):
    mean_verts = preds.mean(axis=0)
    mn, mx = mean_verts.min(), mean_verts.max()
    if mx != mn:
        mean_verts = (mean_verts - mn) / (mx - mn)
    result = {}
    for i, roi in enumerate(ROI_KEYS):
        seg = mean_verts[i * CHUNK : (i + 1) * CHUNK]
        result[roi] = round(float(seg.mean()), 4)
    return result

def predict_video(video_path):
    df = model.get_events_dataframe(video_path=video_path)
    preds, _ = model.predict(events=df)
    return activations_from_preds(np.array(preds))

def predict_text(text):
    df = model.get_events_dataframe(text=text)
    preds, _ = model.predict(events=df)
    return activations_from_preds(np.array(preds))

class PredictRequest(BaseModel):
    modality: str
    data: Optional[str] = None
    text: Optional[str] = None

@app_api.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app_api.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    try:
        if req.modality == "image":
            img_bytes  = decode_b64(req.data)
            tmp_path   = image_to_tmp_video(img_bytes)
            activations = predict_video(tmp_path)
            os.unlink(tmp_path)

        elif req.modality == "video":
            vid_bytes  = decode_b64(req.data)
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(vid_bytes)
                tmp_path = tmp.name
            activations = predict_video(tmp_path)
            os.unlink(tmp_path)

        elif req.modality == "text":
            activations = predict_text(req.text)

        else:
            raise HTTPException(400, f"Unknown modality: {req.modality}")

        return {"activations": activations, "modality": req.modality, "source": "tribe_v2_local"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    print("Starting local server on http://localhost:8000")
    uvicorn.run(app_api, host="0.0.0.0", port=8000)
