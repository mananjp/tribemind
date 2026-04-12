"""
TribeMind — Brain Response Visualizer
Streamlit frontend that sends media/text to the TRIBE v2 backend
and renders an interactive ROI brain map.
"""
import os, sys, tempfile, json, time, logging
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from PIL import Image
from groq import Groq

log = logging.getLogger(__name__)

# Allow imports from the backend folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from brain_regions import BRAIN_REGIONS, ROI_CATEGORIES, get_region_info
from inference import predict_from_image, predict_from_video, predict_from_text

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TribeMind \u00b7 Brain Response Visualizer",
    page_icon="\U0001f9e0",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1c1f26;
        border-radius: 8px;
        padding: 8px 20px;
        color: #ccc;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    .roi-card {
        background: #1c1f26;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        border-left: 4px solid;
    }
    .metric-box {
        background: #1c1f26;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    h1 { color: #4CAF50; }
    .summary-card {
        background: linear-gradient(135deg, #1a2332 0%, #1c2a3a 50%, #1a2332 100%);
        border: 1px solid #2d4a5e;
        border-radius: 14px;
        padding: 24px 28px;
        margin: 12px 0 20px 0;
        position: relative;
        overflow: hidden;
    }
    .summary-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #4CAF50, #2196F3, #9C27B0, #FF9800);
    }
    .summary-card h4 {
        color: #4CAF50;
        margin: 0 0 14px 0;
        font-size: 1.15em;
    }
    .summary-section {
        margin-bottom: 14px;
    }
    .summary-section .label {
        color: #7eb8da;
        font-weight: 600;
        font-size: 0.9em;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .summary-section .content {
        color: #d4dde8;
        font-size: 0.98em;
        line-height: 1.55;
    }
    .outcome-tag {
        display: inline-block;
        background: rgba(76, 175, 80, 0.15);
        border: 1px solid rgba(76, 175, 80, 0.3);
        color: #81c784;
        border-radius: 20px;
        padding: 4px 14px;
        margin: 3px 4px 3px 0;
        font-size: 0.85em;
    }
    .edu-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #e94560;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 10px 0;
        font-size: 0.88em;
        color: #e0e0e0;
        line-height: 1.5;
    }
    .edu-banner strong { color: #e94560; }
    .neuro-insight {
        background: linear-gradient(135deg, #1a1a2e 0%, #2d1b4e 100%);
        border: 1px solid #7c4dff;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    .neuro-insight .label {
        color: #b388ff;
        font-weight: 600;
        font-size: 0.88em;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .neuro-insight .content {
        color: #d4dde8;
        font-size: 0.95em;
        line-height: 1.55;
    }
    .dopamine-meter {
        background: #1c1f26;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        border-left: 3px solid #E040FB;
    }
    .dopamine-meter .meter-bar {
        background: #2a2a3a;
        border-radius: 4px;
        height: 12px;
        margin-top: 6px;
        overflow: hidden;
    }
    .dopamine-meter .meter-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #7c4dff, #E040FB, #FF4081);
        transition: width 0.5s ease;
    }
    .llm-card {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border: 1px solid #4dd0e1;
        border-radius: 14px;
        padding: 24px 28px;
        margin: 16px 0;
        position: relative;
        overflow: hidden;
    }
    .llm-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #4dd0e1, #E040FB, #FF4081, #FFD54F);
    }
    .llm-card h4 {
        color: #4dd0e1;
        margin: 0 0 14px 0;
        font-size: 1.15em;
    }
    .llm-card .llm-body {
        color: #dce6ed;
        font-size: 0.96em;
        line-height: 1.65;
    }
    .llm-card .llm-body strong { color: #80deea; }
    .llm-card .llm-body em { color: #b0bec5; }
    .llm-card .llm-body h5 { color: #4dd0e1; margin: 12px 0 4px 0; font-size: 1em; }
    .llm-card .llm-footer {
        margin-top: 14px;
        font-size: 0.78em;
        color: #78909c;
        border-top: 1px solid #37474f;
        padding-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Lobes_of_the_brain_NL.svg/600px-Lobes_of_the_brain_NL.svg.png",
             use_container_width=True)
    st.markdown("## \U0001f9e0 TribeMind")
    st.markdown(
        "Powered by **Meta TRIBE v2** \u2014 a tri-modal fMRI foundation model "
        "trained on 500+ hours of brain scans from 700+ volunteers."
    )
    # Ensure backend URL is in env, defaulting to localhost if not set in .env
    if "TRIBE_BACKEND_URL" not in os.environ:
        os.environ["TRIBE_BACKEND_URL"] = "http://localhost:8000"

    st.divider()
    activation_threshold = st.slider("Activation threshold", 0.0, 1.0, 0.3, 0.05)
    top_n = st.slider("Show top N regions", 3, len(BRAIN_REGIONS), 10)

    st.divider()
    st.markdown("### \U0001f393 Research Mode")
    edu_mode = st.toggle("\U0001f52c Educational Neuroscience Mode", value=False,
                         help="Enables detailed reward-circuit analysis, dopamine pathway insights, and neuroscience deep-dives. Intended strictly for academic and educational use.")
    if edu_mode:
        st.info("\U0001f3eb Research mode ON \u2014 showing full neuroscience analysis including reward circuits, arousal pathways, and dopaminergic system insights.")

    st.divider()
    st.markdown("### \U0001f916 AI Summary")
    
    # Read Groq key directly from environment (set via .env)
    groq_key = os.getenv("GROQ_API_KEY", "")
    groq_model = st.selectbox("LLM Model", [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ], index=0)
    enable_llm = st.toggle("\u2728 Enable AI-powered summary", value=bool(groq_key),
                           help="Uses Groq LLM to generate a personalized, in-depth analysis of the brain activity.")

# ── Helper: generate plain-language summary ──────────────────────────────────
_CATEGORY_EXPLANATIONS = {
    "Vision": {
        "active": "Your visual processing system is highly engaged \u2014 the brain is actively analyzing what it sees, breaking down shapes, colors, motion, and recognizing objects or faces.",
        "meaning": "This stimulus is visually rich and demands significant perceptual processing.",
        "outcomes": ["Heightened visual attention", "Better image recall", "Stronger visual memory formation", "Increased pattern recognition"],
        "commercial": "High visual engagement predicts strong ad recall and brand recognition. This translates to excellent performance for display advertising, packaging design, and visual product demonstrations.",
        "personal": "You will likely find this visually stimulating and easy to picture later. It captures your gaze and focuses your visual attention seamlessly."
    },
    "Language": {
        "active": "The brain's language centers have fired up \u2014 it's parsing words, constructing meaning from sentences, and working through grammar and semantics.",
        "meaning": "The input contains meaningful linguistic content that the brain is actively decoding.",
        "outcomes": ["Deeper comprehension", "Inner speech activation", "Vocabulary enrichment", "Analytical thinking boost"],
        "commercial": "Strong linguistic engagement means your messaging, slogans, or copy are being deeply processed. Ideal for detailed pitches, educational content, or thought leadership.",
        "personal": "This content is sparking your analytical mind and inner voice. You are actively 'talking to yourself' as you process the core information."
    },
    "Auditory": {
        "active": "Auditory processing areas are active \u2014 the brain is interpreting sounds, distinguishing tones, and processing rhythm or speech patterns.",
        "meaning": "Sound-related content is being processed even if the input is visual (the brain may be 'hearing' implied sounds).",
        "outcomes": ["Sound awareness", "Rhythmic processing", "Speech perception enhancement", "Musical sensitivity"],
        "commercial": "Auditory triggers are powerful for creating sticky jingles, sonic branding (brand sounds), and maintaining attention in multimedia formats like podcasts or video ads.",
        "personal": "You are becoming more attuned to the rhythms, voices, or ambient sounds here, which can strongly affect your mood and arousal level."
    },
    "Social": {
        "active": "Social cognition circuits are engaged \u2014 the brain is trying to understand other people's intentions, emotions, and social dynamics.",
        "meaning": "This content involves people, relationships, or social situations that trigger empathy and perspective-taking.",
        "outcomes": ["Empathy activation", "Theory of mind processing", "Social judgment", "Moral reasoning"],
        "commercial": "Social activation builds brand trust and relatability. It indicates that characters, influencers, or user testimonials are successfully resonating with the audience.",
        "personal": "You're relating to the humans in this experience. This triggers your empathy, making you care more about the people involved and their outcomes."
    },
    "Emotion": {
        "active": "The brain's emotional processing hub (amygdala) is responding \u2014 something in this stimulus triggers an emotional reaction, whether excitement, fear, or surprise.",
        "meaning": "The content has emotional salience \u2014 it grabs attention and creates a visceral response.",
        "outcomes": ["Emotional arousal", "Enhanced memory encoding", "Fight-or-flight assessment", "Mood shift"],
        "commercial": "A high emotional response is the #1 predictor of virality and long-term brand loyalty. Content that makes people feel something drives the highest conversion rates.",
        "personal": "This is striking a chord with you. Whether it's joy, nostalgia, or tension, your body is physically reacting to the emotional weight of this content."
    },
    "Memory": {
        "active": "Memory systems are actively engaged \u2014 the hippocampus is forming new memories or retrieving old ones, connecting this experience to what you already know.",
        "meaning": "This stimulus activates prior knowledge or creates strong enough impressions to be stored as long-term memories.",
        "outcomes": ["Memory consolidation", "Spatial navigation thoughts", "Nostalgia or familiarity", "Learning reinforcement"],
        "commercial": "Memory activation is crucial for brand longevity. This content is either successfully invoking nostalgia or is distinctive enough to be cemented into long-term recall.",
        "personal": "You are connecting this to your own past experiences. It feels familiar, nostalgic, or deeply informative, helping you learn and retain the information."
    },
    "Reward": {
        "active": "The brain's reward circuitry (mesolimbic dopamine system) is engaged \u2014 the Nucleus Accumbens, VTA, and Orbitofrontal Cortex are processing pleasure, motivation, and subjective value.",
        "meaning": "This stimulus triggers the brain's wanting/liking systems, activating dopaminergic pathways that govern motivation, craving, and reinforcement.",
        "outcomes": ["Dopamine release", "Motivation & wanting response", "Reinforcement learning", "Approach behavior", "Physiological arousal", "Hedonic evaluation"],
        "commercial": "Reward-circuit activation is the neurological basis of product desire, purchase intent, and brand craving. This is the neural signature behind impulse buying, subscription retention, and viral sharing.",
        "personal": "Your brain is assigning high subjective value to this \u2014 it feels rewarding, pleasurable, or deeply engaging. The dopamine system is reinforcing your attention and motivating continued engagement.",
        "neuroscience": "The mesolimbic dopamine pathway (VTA \u2192 NAcc) is the brain's core reward circuit, evolutionarily designed to reinforce survival behaviors. When activated by modern stimuli (food imagery, social validation, attractive faces, novel content), it produces the subjective experience of 'wanting'. The Orbitofrontal Cortex (OFC) computes the hedonic value, while the Insular Cortex monitors the body's physiological response. The Anterior Cingulate Cortex (ACC) handles the cognitive-emotional conflict, important when content triggers competing reactions (attraction vs. social norms). Research shows these circuits respond similarly across reward types \u2014 food, social, monetary, and sensory rewards all converge on this pathway (Berridge & Kringelbach, 2015)."
    },
}

_COMBINED_PATTERNS = {
    ("Vision", "Emotion"): "When vision and emotion fire together, it means the content is not just seen \u2014 it's <em>felt</em>. The brain tags this experience as emotionally significant, making it far more likely to be remembered.",
    ("Vision", "Social"): "The brain is not only processing visual content but also reading social cues from it \u2014 analyzing body language, facial expressions, or interpersonal dynamics visible in the scene.",
    ("Language", "Social"): "Language and social systems working together means the brain is processing narrative content with social meaning \u2014 understanding characters, motivations, or persuasive arguments.",
    ("Language", "Memory"): "When language meets memory, the brain is connecting words to personal experiences \u2014 this content may remind you of something or teach you something that 'sticks'.",
    ("Vision", "Memory"): "Visual content is triggering memory circuits \u2014 you may be recognizing familiar places, objects, or scenes, or the brain is encoding this visual experience for future recall.",
    ("Emotion", "Memory"): "An emotionally charged memory response \u2014 the brain is creating vivid, lasting memories. Emotional experiences are remembered 2-3x better than neutral ones.",
    ("Emotion", "Social"): "Social-emotional processing is at work \u2014 the brain is assessing emotional dynamics between people, which drives empathy, trust judgments, and social bonding.",
    ("Vision", "Language"): "The brain is simultaneously processing visual information and extracting linguistic/semantic meaning \u2014 this could be reading text in images, or the visual content is so meaningful the brain 'narrates' it internally.",
    ("Vision", "Reward"): "Visual content is directly activating the reward system \u2014 what's being seen is perceived as inherently valuable or pleasurable. This is the neural basis of visual appeal and aesthetic attraction.",
    ("Emotion", "Reward"): "Emotion and reward are deeply intertwined \u2014 the brain is not just feeling something, it's <em>wanting</em> more. This dual activation drives compulsive engagement, strong approach motivation, and heightened arousal.",
    ("Social", "Reward"): "Social-reward co-activation represents the brain processing social content as intrinsically rewarding \u2014 the neural basis of social bonding, attraction, and the 'like' button psychology.",
    ("Memory", "Reward"): "When reward circuits fire alongside memory systems, the brain is cementing this as a high-value experience. This creates powerful associative memories \u2014 the kind that drive nostalgia-based cravings and brand loyalty.",
    ("Reward", "Language"): "Language is being processed in a reward-laden context \u2014 the words themselves carry motivational weight, creating persuasion, desire, or intellectual pleasure.",
    ("Reward", "Vision"): "Visual content is directly activating the reward system \u2014 what's being seen is perceived as inherently valuable or pleasurable. This is the neural basis of visual appeal and aesthetic attraction.",
    ("Reward", "Emotion"): "Emotion and reward are deeply intertwined \u2014 the brain is not just feeling something, it's <em>wanting</em> more. This dual activation drives compulsive engagement, strong approach motivation, and heightened arousal.",
    ("Reward", "Social"): "Social-reward co-activation represents the brain processing social content as intrinsically rewarding \u2014 the neural basis of social bonding, attraction, and the 'like' button psychology.",
    ("Reward", "Memory"): "When reward circuits fire alongside memory systems, the brain is cementing this as a high-value experience. This creates powerful associative memories \u2014 the kind that drive nostalgia-based cravings and brand loyalty.",
}


def _generate_summary(activations: dict, threshold: float, edu_mode: bool = False) -> dict:
    """Generate a plain-language summary of what the brain activity means."""
    # Compute per-category scores
    cat_scores = {}
    cat_regions = {}
    for roi, val in activations.items():
        info = BRAIN_REGIONS.get(roi, {})
        cat = info.get("category", "Other")
        cat_scores[cat] = cat_scores.get(cat, 0) + val
        if val >= threshold:
            cat_regions.setdefault(cat, []).append((roi, val, info))

    # Sort categories by total activation
    sorted_cats = sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)
    active_cats = [c for c, s in sorted_cats if c in cat_regions and c != "Other"]

    # Top 3 regions overall
    top3 = sorted(activations.items(), key=lambda x: x[1], reverse=True)[:3]

    # Build the "what's happening" paragraph
    happening_parts = []
    commercial_parts = []
    personal_parts = []
    neuroscience_parts = []
    for cat in active_cats[:3]:  # describe top 3 systems
        exp = _CATEGORY_EXPLANATIONS.get(cat)
        if exp:
            happening_parts.append(exp["active"])
            if "commercial" in exp:
                commercial_parts.append(exp["commercial"])
            if "personal" in exp:
                personal_parts.append(exp["personal"])
            if edu_mode and "neuroscience" in exp:
                neuroscience_parts.append(exp["neuroscience"])

    # Build combined-pattern insight
    combined_insight = ""
    if len(active_cats) >= 2:
        pair = tuple(active_cats[:2])
        combined_insight = _COMBINED_PATTERNS.get(pair, "")
        if not combined_insight:
            combined_insight = _COMBINED_PATTERNS.get((pair[1], pair[0]), "")

    # Build the "top regions" plain-language sentence
    top_sentences = []
    for roi, val, *_ in top3:
        info = BRAIN_REGIONS.get(roi, {})
        name = info.get("full_name", roi)
        func = info.get("function", "")
        pct = int(val * 100)
        top_sentences.append("<strong>" + name + "</strong> (" + str(pct) + "%) \u2014 " + func.rstrip("."))

    # Collect outcomes
    outcomes = []
    for cat in active_cats[:4]:
        exp = _CATEGORY_EXPLANATIONS.get(cat)
        if exp:
            outcomes.extend(exp["outcomes"][:2])  # top 2 per category

    # Overall intensity level
    avg = sum(activations.values()) / max(len(activations), 1)
    if avg > 0.5:
        intensity = "\U0001f534 <strong>High intensity</strong> \u2014 This stimulus strongly engages the brain across multiple systems."
    elif avg > 0.3:
        intensity = "\U0001f7e1 <strong>Moderate intensity</strong> \u2014 A solid level of brain engagement, with clear focal areas of activity."
    else:
        intensity = "\U0001f7e2 <strong>Low intensity</strong> \u2014 The brain is responding mildly; this stimulus is relatively neutral."

    # Reward / dopamine metrics (for educational mode)
    reward_score = 0.0
    reward_regions_active = []
    reward_rois = ["NAcc", "OFC", "Insula", "ACC", "VTA", "Hypothalamus"]
    for roi in reward_rois:
        v = activations.get(roi, 0)
        reward_score += v
        if v >= threshold:
            info = BRAIN_REGIONS.get(roi, {})
            reward_regions_active.append((roi, v, info.get("full_name", roi), info.get("function", "")))
    reward_score = reward_score / max(len(reward_rois), 1)

    # Attention economy score
    attention_rois = ["V1", "FFA", "Amygdala", "NAcc", "mPFC"]
    attention_score = sum(activations.get(r, 0) for r in attention_rois) / max(len(attention_rois), 1)

    # Memorability score
    memory_rois = ["Hippocampus", "Amygdala", "NAcc", "RSC"]
    memorability_score = sum(activations.get(r, 0) for r in memory_rois) / max(len(memory_rois), 1)

    # Engagement profile label
    if attention_score > 0.6 and reward_score > 0.5:
        engagement_label = "\U0001f525 Highly Captivating"
        engagement_detail = "This content commands attention and activates reward circuits simultaneously \u2014 the hallmark of deeply engaging, hard-to-ignore stimuli."
    elif attention_score > 0.5:
        engagement_label = "\U0001f441\ufe0f Attention-Grabbing"
        engagement_detail = "This content captures visual and cognitive focus strongly, even if the reward response is moderate."
    elif reward_score > 0.5:
        engagement_label = "\U0001f9e0 Intrinsically Rewarding"
        engagement_detail = "The brain finds this content pleasurable or motivating, even without extreme visual salience."
    elif memorability_score > 0.5:
        engagement_label = "\U0001f4be Memory-Forming"
        engagement_detail = "This stimulus is being encoded into long-term memory \u2014 it will likely be recalled later."
    else:
        engagement_label = "\u26aa Neutral"
        engagement_detail = "The brain's response is balanced and mild \u2014 no single system dominates strongly."

    return {
        "happening": " ".join(happening_parts),
        "commercial": " ".join(commercial_parts),
        "personal": " ".join(personal_parts),
        "neuroscience": " ".join(neuroscience_parts),
        "combined_insight": combined_insight,
        "top_regions": top_sentences,
        "outcomes": outcomes,
        "intensity": intensity,
        "dominant_systems": active_cats[:4],
        "reward_score": reward_score,
        "reward_regions_active": reward_regions_active,
        "attention_score": attention_score,
        "memorability_score": memorability_score,
        "engagement_label": engagement_label,
        "engagement_detail": engagement_detail,
        "edu_mode": edu_mode,
    }


# ── Helper: draw radar chart ──────────────────────────────────────────────────
def _radar(activations: dict, threshold: float) -> go.Figure:
    rois  = list(activations.keys())
    vals  = [activations[r] for r in rois]
    colors = [BRAIN_REGIONS[r]["color"] if r in BRAIN_REGIONS else "#9E9E9E" for r in rois]
    rois_c = rois + [rois[0]]
    vals_c = vals  + [vals[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vals_c, theta=rois_c,
        fill="toself",
        fillcolor="rgba(76,175,80,0.15)",
        line=dict(color="#4CAF50", width=2),
        marker=dict(size=6, color=colors)
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(visible=True, range=[0, 1], color="#666", gridcolor="#333"),
            angularaxis=dict(color="#aaa", gridcolor="#333")
        ),
        paper_bgcolor="#0e1117", font_color="#eee",
        margin=dict(l=40, r=40, t=40, b=40), height=420
    )
    return fig


# ── Helper: horizontal bar chart ─────────────────────────────────────────────
def _bar(activations: dict, top_n: int) -> go.Figure:
    sorted_items = sorted(activations.items(), key=lambda x: x[1], reverse=True)[:top_n]
    rois, vals   = zip(*sorted_items)
    colors = [BRAIN_REGIONS[r]["color"] if r in BRAIN_REGIONS else "#9E9E9E" for r in rois]
    fig = go.Figure(go.Bar(
        x=list(vals), y=list(rois), orientation="h",
        marker_color=colors,
        text=[f"{v:.0%}" for v in vals],
        textposition="outside"
    ))
    fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#161b22",
        font_color="#eee", height=380,
        xaxis=dict(range=[0, 1.15], title="Activation strength", color="#aaa", gridcolor="#333"),
        yaxis=dict(autorange="reversed", color="#aaa"),
        margin=dict(l=10, r=40, t=20, b=30)
    )
    return fig


# ── Helper: category donut ────────────────────────────────────────────────────
def _donut(activations: dict) -> go.Figure:
    cat_scores: dict = {}
    for roi, val in activations.items():
        cat = BRAIN_REGIONS.get(roi, {}).get("category", "Other")
        cat_scores[cat] = cat_scores.get(cat, 0) + val
    labels = list(cat_scores.keys())
    values = [cat_scores[l] for l in labels]
    colors = [ROI_CATEGORIES.get(l, "#9E9E9E") for l in labels]
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.52, marker_colors=colors,
        textinfo="label+percent",
        insidetextorientation="radial"
    ))
    fig.update_layout(
        paper_bgcolor="#0e1117", font_color="#eee",
        showlegend=False, height=320,
        margin=dict(l=10, r=10, t=20, b=10)
    )
    return fig


def _build_summary_html(summary: dict) -> str:
    """Build the HTML for the plain-language summary card."""
    parts = []
    parts.append('<div class="summary-card">')
    parts.append('<h4>\U0001f4a1 In Simple Terms \u2014 What This Brain Activity Means</h4>')

    # Intensity
    parts.append('<div class="summary-section">')
    parts.append('<div class="label">\u26a1 Overall Intensity</div>')
    parts.append('<div class="content">' + summary["intensity"] + '</div>')
    parts.append('</div>')

    # Engagement profile
    parts.append('<div class="summary-section">')
    parts.append('<div class="label">\U0001f4ca Engagement Profile</div>')
    parts.append('<div class="content"><strong>' + summary["engagement_label"] + '</strong> \u2014 ' + summary["engagement_detail"] + '</div>')
    parts.append('</div>')

    # What's happening
    parts.append('<div class="summary-section">')
    parts.append('<div class="label">\U0001f9e0 What Happens in the Brain</div>')
    parts.append('<div class="content">' + summary["happening"] + '</div>')
    parts.append('</div>')

    # Commercial & Personal Insights
    if summary.get("commercial"):
        parts.append('<div class="summary-section">')
        parts.append('<div class="label">\U0001f4bc Commercial & Business Value</div>')
        parts.append('<div class="content">' + summary["commercial"] + '</div>')
        parts.append('</div>')

    if summary.get("personal"):
        parts.append('<div class="summary-section">')
        parts.append('<div class="label">\U0001f464 Personal Experience</div>')
        parts.append('<div class="content">' + summary["personal"] + '</div>')
        parts.append('</div>')

    # Cross-system insight
    if summary["combined_insight"]:
        parts.append('<div class="summary-section">')
        parts.append('<div class="label">\U0001f517 Cross-system Insight</div>')
        parts.append('<div class="content">' + summary["combined_insight"] + '</div>')
        parts.append('</div>')

    # Top regions
    region_items = "".join("<li>" + s + "</li>" for s in summary["top_regions"])
    parts.append('<div class="summary-section">')
    parts.append('<div class="label">\U0001f4cd Most Active Regions</div>')
    parts.append('<div class="content"><ol style="margin:4px 0 0 18px;padding:0;">' + region_items + '</ol></div>')
    parts.append('</div>')

    # Outcomes
    outcome_tags = "".join('<span class="outcome-tag">' + o + '</span>' for o in summary["outcomes"])
    parts.append('<div class="summary-section">')
    parts.append('<div class="label">\U0001f3af Possible Outcomes &amp; Behaviors</div>')
    parts.append('<div class="content">' + outcome_tags + '</div>')
    parts.append('</div>')

    # Composite scores
    att_pct = int(summary["attention_score"] * 100)
    mem_pct = int(summary["memorability_score"] * 100)
    rew_pct = int(summary["reward_score"] * 100)
    parts.append('<div class="summary-section">')
    parts.append('<div class="label">\U0001f4c8 Composite Neuro-Scores</div>')
    parts.append('<div class="content">')
    parts.append('<div class="dopamine-meter"><strong>\U0001f440 Attention Capture:</strong> ' + str(att_pct) + '%<div class="meter-bar"><div class="meter-fill" style="width:' + str(att_pct) + '%"></div></div></div>')
    parts.append('<div class="dopamine-meter"><strong>\U0001f4be Memorability:</strong> ' + str(mem_pct) + '%<div class="meter-bar"><div class="meter-fill" style="width:' + str(mem_pct) + '%"></div></div></div>')
    parts.append('<div class="dopamine-meter"><strong>\U0001f49c Reward Activation:</strong> ' + str(rew_pct) + '%<div class="meter-bar"><div class="meter-fill" style="width:' + str(rew_pct) + '%"></div></div></div>')
    parts.append('</div></div>')

    parts.append('</div>')  # close summary-card

    # Educational neuroscience deep-dive (only in edu mode)
    if summary.get("edu_mode") and (summary.get("neuroscience") or summary.get("reward_regions_active")):
        parts.append('<div class="neuro-insight">')
        parts.append('<div class="label">\U0001f52c Neuroscience Deep-Dive (Educational)</div>')

        if summary.get("neuroscience"):
            parts.append('<div class="content" style="margin-bottom:12px;">' + summary["neuroscience"] + '</div>')

        if summary.get("reward_regions_active"):
            parts.append('<div class="content"><strong>Active reward-circuit regions:</strong></div>')
            for roi, val, name, func in summary["reward_regions_active"]:
                pct = int(val * 100)
                parts.append('<div class="dopamine-meter"><strong>' + name + '</strong> (' + str(pct) + '%) \u2014 ' + func + '</div>')

        parts.append('<div class="content" style="margin-top:12px;font-size:0.82em;color:#999;">'
                     '<em>\U0001f4d6 References: Berridge & Kringelbach (2015) Pleasure Systems in the Brain; '
                     'Schultz (2015) Neuronal Reward and Decision Signals; '
                     'Haber & Knutson (2010) The Reward Circuit. '
                     'This analysis is for educational and research purposes only.</em></div>')
        parts.append('</div>')

    return "".join(parts)


# ── Helper: Groq LLM-powered summary ─────────────────────────────────────────
def _generate_llm_summary(
    activations: dict,
    summary: dict,
    modality: str,
    edu_mode: bool,
    groq_key: str,
    groq_model: str,
    stimulus_content: str,
) -> str:
    """Call Groq LLM to generate a personalized, context-aware brain analysis."""
    try:
        client = Groq(api_key=groq_key)

        # Build activation table for the prompt
        act_lines = []
        for roi, val in sorted(activations.items(), key=lambda x: x[1], reverse=True):
            info = BRAIN_REGIONS.get(roi, {})
            pct = int(val * 100)
            act_lines.append(f"  {info.get('full_name', roi)} ({roi}): {pct}% — {info.get('function', 'N/A')} [{info.get('category', 'Other')}]")
        activation_table = "\n".join(act_lines)

        # Composite scores
        att_pct = int(summary["attention_score"] * 100)
        mem_pct = int(summary["memorability_score"] * 100)
        rew_pct = int(summary["reward_score"] * 100)

        # Active systems
        dominant = ", ".join(summary.get("dominant_systems", []))

        # Educational context
        edu_context = ""
        if edu_mode:
            edu_context = """
EDUCATIONAL MODE IS ACTIVE. The user has opted into full neuroscience educational mode. You MUST:
- Explicitly connect the CONTENT of the user's input (if provided) to the resulting brain activations.
- If the text implies a specific intention (e.g., desperation, lust, hunger, social craving), explain precisely HOW and WHY those intentions trigger specific circuits (like physiological arousal in the Insula, reward anticipation in the VTA/NAcc, or evaluation in the OFC).
- Discuss the evolutionary purpose of these activations in relation to the specific input text.
- Reference relevant neuroscience concepts (dopamine, serotonin, mirror neurons, etc.) objectively.
- All content is for EDUCATIONAL PURPOSES. Do not censor or shy away from explaining brain mechanisms related to explicit or sensitive stimulus content — explain the science objectively, frankly, and thoroughly.
"""

        system_prompt = f"""You are a world-class neuroscientist and science communicator for TribeMind — a brain response visualization tool powered by Meta's TRIBE v2 fMRI model.

Your job is to produce a PERSONALIZED, engaging, and scientifically grounded analysis of the user's brain activation data. Write as if you're explaining directly to the user.

Rules:
1. Be vivid, warm, and engaging — NOT dry or clinical
2. Use analogies and real-world examples to make neuroscience accessible
3. Structure your response with clear sections using <h5> tags
4. Use <strong> for emphasis and <em> for subtlety
5. Output ONLY valid HTML (no markdown). Do NOT wrap in code blocks.
6. Keep the response between 300-500 words
7. Do NOT repeat the raw numbers — interpret them meaningfully
8. End with a practical takeaway: what does this mean for the user personally?
{edu_context}"""

        content_section = f"\nCONTENT/STIMULUS ANALYZED:\n\"{stimulus_content}\"\n" if stimulus_content else ""

        user_prompt = f"""Analyze this brain activation data from a {modality} stimulus:
{content_section}
BRAIN REGION ACTIVATIONS:
{activation_table}

COMPOSITE SCORES:
- Attention Capture: {att_pct}%
- Memorability: {mem_pct}%
- Reward Activation: {rew_pct}%

ENGAGEMENT PROFILE: {summary['engagement_label']}
DOMINANT BRAIN SYSTEMS: {dominant}

Generate a personalized neuroscience analysis. Focus deeply on linking the meaning/intention of the CONTENT (if provided) with what this SPECIFIC pattern of activations reveals about how the brain processes it."""

        response = client.chat.completions.create(
            model=groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=1024,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        log.warning("Groq LLM error: %s", e)
        st.warning(f"\u26a0\ufe0f AI summary failed: {e}")
        return ""


# ── Helper: render results ────────────────────────────────────────────────────
def _render_results(result: dict):
    activations = result["activations"]
    source = result.get("source", "mock")
    if source == "mock":
        source_tag = "\U0001f7e1 Demo mode"
    elif source == "mock_fallback":
        source_tag = "\U0001f7e0 Backend error \u2014 showing demo results"
    else:
        source_tag = "\U0001f7e2 Live (TRIBE v2)"
    st.caption(source_tag)
    if result.get("backend_error"):
        st.warning("\u26a0\ufe0f Backend returned an error: `" + result["backend_error"] + "` \u2014 showing simulated results instead.")

    # ── KPI row ──────────────────────────────────────────────────────────────
    active_rois  = [r for r, v in activations.items() if v >= activation_threshold]
    top_roi      = max(activations, key=activations.get)
    top_cat_raw  = {}
    for roi, val in activations.items():
        cat = BRAIN_REGIONS.get(roi, {}).get("category", "Other")
        top_cat_raw[cat] = top_cat_raw.get(cat, 0) + val
    dominant_cat = max(top_cat_raw, key=top_cat_raw.get)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Active regions",   len(active_rois))
    k2.metric("Peak region",      top_roi)
    k3.metric("Peak activation",  f"{activations[top_roi]:.0%}")
    k4.metric("Dominant system",  dominant_cat)

    st.divider()

    # ── Plain-language summary ────────────────────────────────────────────────
    summary = _generate_summary(activations, activation_threshold, edu_mode=edu_mode)
    st.markdown(_build_summary_html(summary), unsafe_allow_html=True)

    # Educational disclaimer banner (when edu mode is on)
    if edu_mode:
        st.markdown(
            '<div class="edu-banner">'
            '<strong>\U0001f393 Educational Research Mode</strong><br>'
            'The reward-circuit and arousal analysis shown above is grounded in peer-reviewed neuroscience research. '
            'The brain\'s reward system responds to a broad spectrum of stimuli \u2014 from food and music to social validation and explicit content \u2014 '
            'using the same core dopaminergic pathways (Berridge & Kringelbach, 2015). This tool visualizes these responses '
            'for <strong>educational understanding only</strong>. It does not endorse, encourage, or promote any specific type of content consumption.'
            '</div>', unsafe_allow_html=True
        )

    # ── LLM-powered personalized summary ──────────────────────────────────────
    if enable_llm and groq_key:
        with st.spinner("\U0001f916 Generating AI-powered personalized analysis..."):
            llm_text = _generate_llm_summary(
                activations=activations,
                summary=summary,
                modality=result.get("modality", "unknown"),
                edu_mode=edu_mode,
                groq_key=groq_key,
                groq_model=groq_model,
                stimulus_content=result.get("stimulus_content", ""),
            )
        if llm_text:
            st.markdown(
                '<div class="llm-card">'
                '<h4>\U0001f916 AI-Powered Personalized Analysis</h4>'
                '<div class="llm-body">' + llm_text + '</div>'
                '<div class="llm-footer">Generated by <strong>' + groq_model + '</strong> via Groq &middot; For educational and research purposes only</div>'
                '</div>',
                unsafe_allow_html=True
            )

    # ── Charts ────────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("### \U0001f578\ufe0f Full-brain radar")
        st.plotly_chart(_radar(activations, activation_threshold), use_container_width=True)
    with col_right:
        st.markdown("### \U0001f4ca System breakdown")
        st.plotly_chart(_donut(activations), use_container_width=True)

    st.markdown("### \U0001f4c8 Top activated regions")
    st.plotly_chart(_bar(activations, top_n), use_container_width=True)

    # ── Region cards ─────────────────────────────────────────────────────────
    st.markdown("### \U0001f50d What each activated region means")
    sorted_rois = sorted(activations.items(), key=lambda x: x[1], reverse=True)
    for roi, val in sorted_rois[:top_n]:
        if val < activation_threshold:
            continue
        info = get_region_info(roi)
        pct  = int(val * 100)
        bar  = "\u2588" * (pct // 5) + "\u2591" * (20 - pct // 5)
        card_html = (
            '<div class="roi-card" style="border-color:' + info["color"] + '">'
            '<b style="color:' + info["color"] + '">' + info["full_name"] + '</b>'
            '&nbsp;&nbsp;<span style="color:#888;font-size:0.85em">(' + info["category"] + ')</span>'
            '&nbsp;&nbsp;<code style="font-size:0.85em">' + bar + ' ' + str(pct) + '%</code>'
            '<br><span style="color:#ddd">' + info["function"] + '</span>'
            '<br><span style="color:#888;font-size:0.82em">Triggered by: ' + ", ".join(info["stimulated_by"][:3]) + '</span>'
            '</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

    # ── Raw JSON expander ────────────────────────────────────────────────────
    with st.expander("\U0001f4c4 Raw prediction JSON"):
        st.json(result)


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("\U0001f9e0 TribeMind \u2014 Brain Response Visualizer")
st.markdown(
    "Upload a **photo**, **video**, or enter **text** to see which brain regions "
    "would light up in an fMRI scan, and what each activation means."
)

tab_img, tab_vid, tab_txt = st.tabs(["\U0001f5bc\ufe0f Image", "\U0001f3ac Video", "\U0001f4dd Text"])

# ── Image tab ─────────────────────────────────────────────────────────────────
with tab_img:
    uploaded_img = st.file_uploader(
        "Upload an image (jpg / png / webp)", type=["jpg", "jpeg", "png", "webp"]
    )
    if uploaded_img:
        col_prev, col_space = st.columns([1, 2])
        with col_prev:
            st.image(uploaded_img, caption="Your image", use_container_width=True)
        if st.button("\U0001f9e0 Predict brain response", key="btn_img"):
            with st.spinner("Calling TRIBE v2 backend\u2026"):
                uploaded_img.seek(0)
                result = predict_from_image(uploaded_img.read())
                result["stimulus_content"] = "[Visual image stimulus uploaded by user]"
            _render_results(result)

# ── Video tab ─────────────────────────────────────────────────────────────────
with tab_vid:
    uploaded_vid = st.file_uploader(
        "Upload a short video (mp4 / webm, \u2264 60 s recommended)", type=["mp4", "webm", "mov"]
    )
    if uploaded_vid:
        st.video(uploaded_vid)
        if st.button("\U0001f9e0 Predict brain response", key="btn_vid"):
            with st.spinner("Sending video to backend (may take ~30 s)\u2026"):
                uploaded_vid.seek(0)
                result = predict_from_video(uploaded_vid.read())
                result["stimulus_content"] = "[Video stimulus uploaded by user]"
            _render_results(result)

# ── Text tab ──────────────────────────────────────────────────────────────────
with tab_txt:
    sample_texts = [
        "Select a sample\u2026",
        "The sunset painted the sky in brilliant shades of orange and pink over the mountains.",
        "Researchers discovered a new species of deep-sea fish with bioluminescent patterns.",
        "She smiled warmly and reached out her hand in greeting.",
        "The algorithm recursively partitions the data until each leaf node is pure.",
    ]
    sample = st.selectbox("Or pick a sample:", sample_texts)
    user_text = st.text_area(
        "Enter any text:", value="" if sample == sample_texts[0] else sample, height=120
    )
    if st.button("\U0001f9e0 Predict brain response", key="btn_txt") and user_text.strip():
        with st.spinner("Running text through TRIBE v2\u2026"):
            result = predict_from_text(user_text.strip())
            result["stimulus_content"] = user_text.strip()
        _render_results(result)
    

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "TribeMind uses [Meta TRIBE v2](https://ai.meta.com/blog/tribe-v2-brain-predictive-foundation-model/) "
    "\u2014 a foundation model trained on 500 h of fMRI data from 700+ subjects. "
    "Results are **predictions** from a computational model, not actual brain scans. "
    "**Not for clinical or diagnostic use.**"
)
st.caption(
    "\U0001f393 **Educational Disclaimer:** The reward-circuit and neuroscience insights provided in Research Mode "
    "are derived from established neuroscience literature (Berridge & Kringelbach, 2015; Schultz, 2015; Haber & Knutson, 2010). "
    "All content analysis \u2014 including explicit or sensitive material \u2014 is presented purely for "
    "**academic understanding of brain function**. This tool does not store, judge, or report any uploaded content. "
    "Users are responsible for ensuring compliance with applicable laws and institutional policies."
)
