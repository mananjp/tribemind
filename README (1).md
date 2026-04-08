# 🧠 TribeMind — Brain Response Visualizer

Predicts which brain regions activate in response to photos, videos, and text
using **Meta's TRIBE v2** fMRI foundation model.

---

## Architecture (no local 9 GB download)

```
[Your browser]
      │ upload media / text
      ▼
[Streamlit frontend]  ─── REST call ───►  [FastAPI backend on Google Colab]
  frontend/app.py                          colab/colab_server.ipynb
                                                │
                                                ▼
                                         [TRIBE v2 on Colab T4 GPU]
                                          facebook/tribev2 on HF Hub
```

The **model lives in Colab** — streamed to Colab's disk via Hugging Face Hub
shards. Your machine only runs the Streamlit UI.

---

## Quick Start

### Step 1 — Start the Colab backend
1. Open `colab/colab_server.ipynb` in Google Colab (GPU runtime).
2. Paste your free [ngrok token](https://dashboard.ngrok.com/authtokens) in Cell 7.
3. Run all cells. Copy the `ngrok` URL printed in Cell 7.

### Step 2 — Run the Streamlit app locally
```bash
pip install -r requirements.txt
streamlit run frontend/app.py
```
Paste the ngrok URL into the sidebar → you're live.

### Demo mode (no Colab)
Leave the Backend URL blank — the app uses simulated activations so you can
explore the UI without any running backend.

---

## File structure
```
tribemind/
├── frontend/
│   └── app.py               ← Streamlit UI
├── backend/
│   ├── brain_regions.py     ← ROI → function lookup (17 regions)
│   └── inference.py         ← Calls remote backend or returns mock
├── colab/
│   └── colab_server.ipynb   ← Full model server (runs on Colab GPU)
├── requirements.txt
└── README.md
```

---

## Brain regions covered (17 ROIs)
| Region | System | What triggers it |
|--------|--------|-----------------|
| V1 / V2 / V3 | Vision | Any visual input |
| FFA | Vision | Faces |
| PPA | Vision | Places & scenes |
| EBA | Vision | Body images |
| LOC | Vision | Objects |
| MT/V5 | Vision | Motion |
| A1 / Belt | Auditory | Sound & speech |
| Broca | Language | Grammar & production |
| Wernicke | Language | Comprehension |
| VWFA | Language | Written words |
| TPJ | Social | Social scenes, empathy |
| Amygdala | Emotion | Emotional salience |
| mPFC | Social | Self-relevant content |
| Hippocampus | Memory | Familiar places/events |
| RSC | Memory | Spatial context |

---

## Limitations
- Predictions are from a computational model, not real fMRI scans.
- TRIBE v2 was trained on video+audio+text — still-image inputs are treated
  as single-frame video clips.
- Not intended for clinical or diagnostic use.
