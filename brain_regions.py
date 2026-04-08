"""Brain region ROI → function lookup table."""

BRAIN_REGIONS = {
    # ── Visual cortex ────────────────────────────────────────────────────────
    "V1": {
        "full_name": "Primary Visual Cortex",
        "function": "Processes raw visual input: edges, contrast, and orientation.",
        "stimulated_by": ["Any visual stimulus", "high-contrast images", "moving objects"],
        "color": "#FF4B4B",
        "category": "Vision"
    },
    "V2_V3": {
        "full_name": "Secondary Visual Areas (V2/V3)",
        "function": "Detects visual patterns, depth cues, and motion direction.",
        "stimulated_by": ["Complex patterns", "depth illusions", "optical motion"],
        "color": "#FF7043",
        "category": "Vision"
    },
    "FFA": {
        "full_name": "Fusiform Face Area",
        "function": "Recognizes faces — both human and animal.",
        "stimulated_by": ["Faces", "face-like objects", "portraits", "eye contact"],
        "color": "#E91E63",
        "category": "Vision"
    },
    "PPA": {
        "full_name": "Parahippocampal Place Area",
        "function": "Processes scenes, buildings, and spatial environments.",
        "stimulated_by": ["Landscapes", "rooms", "architecture", "outdoor scenes"],
        "color": "#9C27B0",
        "category": "Vision"
    },
    "EBA": {
        "full_name": "Extrastriate Body Area",
        "function": "Responds to images of the human body (excluding faces).",
        "stimulated_by": ["Body images", "sports", "dance", "action poses"],
        "color": "#673AB7",
        "category": "Vision"
    },
    "LOC": {
        "full_name": "Lateral Occipital Complex",
        "function": "Object recognition — identifies objects regardless of viewpoint.",
        "stimulated_by": ["Objects", "tools", "animals", "product shots"],
        "color": "#3F51B5",
        "category": "Vision"
    },
    "MT_V5": {
        "full_name": "Middle Temporal Area (MT/V5)",
        "function": "Processes visual motion, flow, and speed.",
        "stimulated_by": ["Moving images", "video", "fast-paced cuts", "sports"],
        "color": "#2196F3",
        "category": "Vision"
    },
    # ── Auditory cortex ──────────────────────────────────────────────────────
    "A1": {
        "full_name": "Primary Auditory Cortex",
        "function": "Basic sound processing: frequency, loudness, tone.",
        "stimulated_by": ["Any audio", "tones", "ambient sound"],
        "color": "#00BCD4",
        "category": "Auditory"
    },
    "Belt_Areas": {
        "full_name": "Auditory Belt / Parabelt Areas",
        "function": "Higher-level sound processing: speech sounds, music rhythm.",
        "stimulated_by": ["Speech", "music", "complex soundscapes"],
        "color": "#009688",
        "category": "Auditory"
    },
    # ── Language areas ───────────────────────────────────────────────────────
    "Broca": {
        "full_name": "Broca's Area (IFG)",
        "function": "Language production, syntax processing, and grammar.",
        "stimulated_by": ["Spoken or written sentences", "complex grammar", "syntax"],
        "color": "#4CAF50",
        "category": "Language"
    },
    "Wernicke": {
        "full_name": "Wernicke's Area (STG/MTG)",
        "function": "Language comprehension — understanding spoken/written words.",
        "stimulated_by": ["Readable text", "dialogue", "spoken narration"],
        "color": "#8BC34A",
        "category": "Language"
    },
    "VWFA": {
        "full_name": "Visual Word Form Area",
        "function": "Recognizes written words and letters at a glance.",
        "stimulated_by": ["Text", "written words", "signs", "captions"],
        "color": "#CDDC39",
        "category": "Language"
    },
    # ── Social / emotional ───────────────────────────────────────────────────
    "TPJ": {
        "full_name": "Temporo-Parietal Junction",
        "function": "Social cognition: reading intentions, empathy, and theory of mind.",
        "stimulated_by": ["Social interactions", "facial expressions", "emotional scenes"],
        "color": "#FFC107",
        "category": "Social"
    },
    "Amygdala": {
        "full_name": "Amygdala",
        "function": "Emotional processing — fear, excitement, and emotional salience.",
        "stimulated_by": ["Threatening images", "emotional faces", "suspenseful video"],
        "color": "#FF9800",
        "category": "Emotion"
    },
    "mPFC": {
        "full_name": "Medial Prefrontal Cortex",
        "function": "Self-referential thinking, decision-making, and social judgments.",
        "stimulated_by": ["Moral dilemmas", "relatable content", "first-person narration"],
        "color": "#FF5722",
        "category": "Social"
    },
    # ── Memory / spatial ─────────────────────────────────────────────────────
    "Hippocampus": {
        "full_name": "Hippocampus",
        "function": "Memory encoding, spatial navigation, and scene recall.",
        "stimulated_by": ["Familiar places", "narrative sequences", "maps", "routes"],
        "color": "#795548",
        "category": "Memory"
    },
    "RSC": {
        "full_name": "Retrosplenial Cortex",
        "function": "Landmark-based navigation and contextual scene memory.",
        "stimulated_by": ["Familiar environments", "landmarks", "spatial context"],
        "color": "#607D8B",
        "category": "Memory"
    },
    # ── Reward / motivation / arousal (educational neuroscience) ─────────────
    "NAcc": {
        "full_name": "Nucleus Accumbens",
        "function": "Core of the brain's reward circuit — processes pleasure, motivation, and reinforcement learning.",
        "stimulated_by": ["Rewarding stimuli", "food imagery", "attractive faces", "surprising outcomes", "music climaxes"],
        "color": "#E040FB",
        "category": "Reward"
    },
    "OFC": {
        "full_name": "Orbitofrontal Cortex",
        "function": "Evaluates subjective value and desirability of a stimulus — the brain's 'worth-it' calculator.",
        "stimulated_by": ["Decision-making", "value judgment", "attractive stimuli", "taste", "smell cues"],
        "color": "#CE93D8",
        "category": "Reward"
    },
    "Insula": {
        "full_name": "Insular Cortex",
        "function": "Interoception — awareness of internal body states, disgust, empathy, and somatic arousal.",
        "stimulated_by": ["Visceral content", "pain observation", "arousing stimuli", "body-awareness triggers"],
        "color": "#F48FB1",
        "category": "Reward"
    },
    "ACC": {
        "full_name": "Anterior Cingulate Cortex",
        "function": "Conflict monitoring, error detection, and emotional-cognitive integration.",
        "stimulated_by": ["Moral dilemmas", "conflicting emotions", "taboo content", "uncertain decisions"],
        "color": "#B39DDB",
        "category": "Reward"
    },
    "VTA": {
        "full_name": "Ventral Tegmental Area",
        "function": "Primary dopamine source — drives wanting, craving, and anticipatory excitement.",
        "stimulated_by": ["Novel stimuli", "anticipation of reward", "unexpected pleasure", "addictive cues"],
        "color": "#EA80FC",
        "category": "Reward"
    },
    "Hypothalamus": {
        "full_name": "Hypothalamus",
        "function": "Regulates homeostasis — hunger, thirst, body temperature, and physiological arousal.",
        "stimulated_by": ["Survival-relevant stimuli", "food imagery", "thermal cues", "arousing content"],
        "color": "#F8BBD0",
        "category": "Reward"
    },
}

ROI_CATEGORIES = {
    "Vision":   "#FF4B4B",
    "Auditory": "#00BCD4",
    "Language": "#4CAF50",
    "Social":   "#FFC107",
    "Emotion":  "#FF9800",
    "Memory":   "#795548",
    "Reward":   "#E040FB",
}

def get_region_info(roi_key: str) -> dict:
    return BRAIN_REGIONS.get(roi_key, {
        "full_name": roi_key,
        "function": "Region function not yet catalogued.",
        "stimulated_by": [],
        "color": "#9E9E9E",
        "category": "Other"
    })
