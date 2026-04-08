# 🧠 TribeMind — Brain Response Visualizer

A neuroscience-powered application that predicts which brain regions activate in response to images, videos, or text, using Meta's TRIBE v2 fMRI foundation model. Features AI-powered personalized analysis via Groq LLM.

## ✨ Features

- **Tri-modal input** — Upload images, videos, or enter text
- **23 brain regions** mapped across 7 systems (Vision, Language, Auditory, Social, Emotion, Memory, Reward)
- **Interactive visualizations** — Radar charts, bar charts, donut breakdowns
- **Composite neuro-scores** — Attention Capture, Memorability, Reward Activation
- **AI-powered summaries** — Personalized neuroscience analysis via Groq LLM
- **Educational Research Mode** — Deep-dive into reward circuits, dopamine pathways, and neuroscience
- **Commercial & Personal insights** — Understand content impact from business and personal perspectives

## 🏗️ Architecture

```
┌──────────────────────────────────────────┐
│              Nginx (port 80)             │
│  ┌─────────────────┬──────────────────┐  │
│  │  / → Streamlit  │ /predict → API   │  │
│  │    (port 8501)  │   (port 8000)    │  │
│  └─────────────────┴──────────────────┘  │
│       Supervisord manages all 3          │
└──────────────────────────────────────────┘
```

- **Frontend**: Streamlit app (`app.py`) — UI, visualizations, LLM summaries
- **Backend**: FastAPI server (`server.py`) — TRIBE v2 model inference
- **Proxy**: Nginx — handles CORS, websockets, single-origin routing

## 🚀 Quick Start

### Local Development

```bash
# 1. Clone and setup
git clone <repo-url>
cd tribemind

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
pip install fastapi uvicorn pydantic huggingface_hub transformers scipy opencv-python-headless
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Start backend
python server.py  # Runs on http://localhost:8000

# 6. Start frontend (in another terminal)
streamlit run app.py  # Runs on http://localhost:8501
```

### Docker Deployment

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 2. Build and run
docker compose up --build -d

# App available at http://localhost
```

## 🔧 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Groq API key for AI summaries | _(none)_ |
| `TRIBE_BACKEND_URL` | Backend server URL | `http://localhost:8000` |

## 📚 Educational Disclaimer

The reward-circuit and neuroscience insights provided in Research Mode are derived from established neuroscience literature (Berridge & Kringelbach, 2015; Schultz, 2015; Haber & Knutson, 2010). All content analysis is presented purely for **academic understanding of brain function**.

## 📄 License

For educational and research use only. Powered by [Meta TRIBE v2](https://ai.meta.com/blog/tribe-v2-brain-predictive-foundation-model/).
