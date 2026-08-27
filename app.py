import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ============================================================
# TomatoCare AI — Streamlit Community Cloud App
# Inference preprocessing matches the training notebook:
# RGB -> 224x224 -> CLAHE -> float32 -> /255
# ============================================================

st.set_page_config(
    page_title="TomatoCare AI",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "model.keras"))
IMG_SIZE = (224, 224)

# The notebook's generator creates class_indices in sorted class-name order.
CLASS_NAMES = [
    "Bacterial_spot",
    "Early_blight",
    "Late_blight",
    "Leaf_Miner",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites",
    "Verticillium_wilt",
    "healthy",
]

DISPLAY_NAMES = {
    "Bacterial_spot": "Bacterial Spot",
    "Early_blight": "Early Blight",
    "Late_blight": "Late Blight",
    "Leaf_Miner": "Leaf Miner",
    "Leaf_Mold": "Leaf Mold",
    "Septoria_leaf_spot": "Septoria Leaf Spot",
    "Spider_mites": "Spider Mites",
    "Verticillium_wilt": "Verticillium Wilt",
    "healthy": "Healthy Leaf",
}

DISEASE_INFO = {
    "healthy": (
        "No major disease pattern was detected by the model. "
        "Continue routine monitoring and good crop hygiene."
    ),
    "Bacterial_spot": (
        "Bacterial Spot can produce small dark lesions. "
        "Reduce leaf wetness, avoid overhead watering, and remove severely affected foliage."
    ),
    "Early_blight": (
        "Early Blight often appears as dark target-like lesions, especially on older leaves. "
        "Improve airflow and remove severely affected leaves."
    ),
    "Late_blight": (
        "Late Blight can progress quickly under cool, humid conditions. "
        "Isolate affected plants and consult appropriate agricultural guidance."
    ),
    "Leaf_Miner": (
        "Leaf Miner damage commonly appears as winding tunnels inside leaves. "
        "Inspect affected foliage and manage the pest early."
    ),
    "Leaf_Mold": (
        "Leaf Mold is associated with high humidity and leaf wetness. "
        "Improve ventilation and reduce prolonged leaf moisture."
    ),
    "Septoria_leaf_spot": (
        "Septoria Leaf Spot produces many small circular lesions. "
        "Remove infected leaves and reduce splash water where practical."
    ),
    "Spider_mites": (
        "Spider Mites may cause fine webbing and speckled or yellow foliage. "
        "Inspect leaf undersides and manage mite populations early."
    ),
    "Verticillium_wilt": (
        "Verticillium Wilt can cause progressive yellowing and wilting. "
        "Remove severely affected plants and consider resistant varieties where suitable."
    ),
}


def clahe_preprocessing(img: np.ndarray) -> np.ndarray:
    """Same CLAHE settings used in the training notebook."""
    img = np.clip(img, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img.astype(np.float32)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    arr = np.asarray(image)
    arr = clahe_preprocessing(arr)
    arr = arr / 255.0
    return np.expand_dims(arr.astype(np.float32), axis=0)


@st.cache_resource(show_spinner="Loading DenseNet121 model...")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"'{MODEL_PATH}' was not found. Make sure model.keras is in the repository root."
        )
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def confidence_level(conf: float) -> str:
    if conf >= 0.80:
        return "High confidence"
    if conf >= 0.50:
        return "Moderate confidence"
    return "Low confidence"


def run_prediction(image: Image.Image):
    model = load_model()
    x = preprocess_image(image)
    probs = model.predict(x, verbose=0)[0]

    if len(probs) != len(CLASS_NAMES):
        raise ValueError(
            f"Model returned {len(probs)} outputs, but the app expects "
            f"{len(CLASS_NAMES)} classes."
        )

    top_indices = np.argsort(probs)[::-1][:5]
    top_idx = int(top_indices[0])
    predicted_key = CLASS_NAMES[top_idx]
    confidence = float(probs[top_idx])

    return predicted_key, confidence, probs, top_indices


# ------------------------- CSS ------------------------------

st.markdown(
    """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: transparent !important;}

    .stApp {
        background:
            radial-gradient(circle at 85% 10%, rgba(63, 165, 147, .16), transparent 28%),
            radial-gradient(circle at 10% 90%, rgba(32, 92, 84, .10), transparent 30%),
            #0b111a;
        color: #f4f7f8;
    }

    .block-container {
        max-width: 1220px;
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .topbar {
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:24px;
        padding:14px 18px;
        margin-bottom:22px;
        border:1px solid rgba(255,255,255,.07);
        background:rgba(12,17,27,.86);
        border-radius:12px;
        box-shadow:0 10px 35px rgba(0,0,0,.18);
    }

    .brand {
        font-size:24px;
        font-weight:800;
        white-space:nowrap;
    }

    .brand-green { color:#6dd3c1; }

    .nav {
        display:flex;
        gap:25px;
        color:#d6dce2;
        font-size:14px;
    }

    .nav a {
        color:#d6dce2 !important;
        text-decoration:none;
    }

    .nav a:hover { color:#76d8c5 !important; }

    .hero {
        background:linear-gradient(145deg, rgba(30,40,55,.97), rgba(16,24,35,.97));
        border:1px solid rgba(255,255,255,.06);
        border-radius:18px;
        padding:48px 34px 34px 34px;
        box-shadow:0 18px 55px rgba(0,0,0,.25);
    }

    .eyebrow {
        color:#70d4c1;
        font-weight:800;
        letter-spacing:.08em;
        font-size:13px;
        text-transform:uppercase;
    }

    .hero-title {
        font-size:clamp(42px, 5.7vw, 72px);
        line-height:.98;
        letter-spacing:-.045em;
        font-weight:850;
        margin:13px 0 18px 0;
    }

    .hero-title .green {color:#92ddcf;}
    .hero-copy {
        color:#b4bfcb;
        font-size:17px;
        line-height:1.6;
        max-width:690px;
    }

    .status {
        display:inline-block;
        margin-top:18px;
        padding:8px 13px;
        border-radius:999px;
        border:1px solid rgba(91,207,185,.35);
        background:rgba(57,151,135,.12);
        color:#b8eee4;
        font-size:13px;
        font-weight:650;
    }

    .metric {
        background:linear-gradient(145deg, rgba(61,157,143,.82), rgba(26,83,79,.75));
        border:1px solid rgba(121,225,210,.20);
        border-radius:14px;
        padding:18px 16px;
        min-height:125px;
    }

    .metric-value {font-size:25px; font-weight:850;}
    .metric-label {color:#d0eee8; font-size:13px; margin-top:4px;}
    .metric-copy {color:#b6d2ce; font-size:12px; margin-top:6px;}

    .scan-card {
        background:linear-gradient(145deg, rgba(19,29,41,.98), rgba(12,19,28,.98));
        border:1px solid rgba(94,209,190,.42);
        border-radius:20px;
        padding:22px;
        box-shadow:0 18px 55px rgba(0,0,0,.22);
    }

    .scan-title {
        font-size:24px;
        font-weight:800;
        margin-bottom:3px;
    }

    .muted {color:#aab6c2;}

    .section-kicker {
        color:#72d5c2;
        text-align:center;
        font-weight:800;
        font-size:14px;
        margin-top:28px;
    }

    .section-title {
        text-align:center;
        font-size:34px;
        font-weight:850;
        margin-top:4px;
    }

    .feature {
        background:linear-gradient(145deg, rgba(58,151,137,.82), rgba(28,83,79,.78));
        border:1px solid rgba(121,225,210,.20);
        border-radius:16px;
        padding:22px;
        min-height:180px;
        text-align:center;
    }

    .feature h3 {font-size:18px; margin:7px 0 8px;}
    .feature p {color:#d1e3e0; font-size:14px; line-height:1.45;}

    .result-box {
        background:linear-gradient(145deg, rgba(30,42,57,.98), rgba(14,22,32,.98));
        border:1px solid rgba(255,255,255,.07);
        border-radius:18px;
        padding:22px;
        margin-top:16px;
    }

    .result-name {font-size:31px; font-weight:850; margin-bottom:2px;}
    .result-conf {color:#7bd9c7; font-size:18px; font-weight:750;}
    .result-desc {color:#b7c1cc; line-height:1.55; margin-top:10px;}

    .footer {
        border-top:1px solid rgba(255,255,255,.10);
        text-align:center;
        color:#8794a3;
        padding:25px 0 5px;
        margin-top:55px;
        font-size:13px;
        line-height:1.6;
    }

    div[data-testid="stFileUploader"] {
        background:rgba(8,13,20,.55);
        border:1.5px dashed rgba(94,208,189,.45);
        border-radius:16px;
        padding:10px;
    }

    .stButton > button {
        border-radius:10px !important;
        min-height:45px !important;
        font-weight:750 !important;
    }

    .stProgress > div > div > div > div {
        background-color:#55c6b2;
    }

    @media (max-width: 800px) {
        .nav {display:none;}
        .hero {padding:30px 22px;}
    }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------- NAVBAR ----------------------------

st.markdown(
    """
<div class="topbar">
    <div class="brand">🌿 <span class="brand-green">TomatoCare</span> AI</div>
    <div class="nav">
        <a href="#home">Home</a>
        <a href="#about">About</a>
        <a href="#how-it-works">How It Works</a>
        <a href="#dataset">Dataset</a>
        <a href="#team">Team</a>
        <a href="#contact">Contact</a>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ------------------------- HERO ------------------------------

st.markdown('<div id="home"></div>', unsafe_allow_html=True)

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown(
        """
<div class="hero">
    <div class="eyebrow">AI-Powered Tomato Leaf Diagnosis</div>
    <div class="hero-title">
        Diagnose Tomato<br>
        <span class="green">Diseases.</span> Protect<br>
        Your Harvest.
    </div>
    <div class="hero-copy">
        TomatoCare AI uses a trained DenseNet121 deep learning model to
        classify tomato leaf images and provide fast, interpretable
        confidence scores for the top predictions.
    </div>
    <div class="status">✓ DenseNet121 model ready for diagnosis</div>
</div>
""",
        unsafe_allow_html=True,
    )

with right:
    st.markdown('<div class="scan-card">', unsafe_allow_html=True)
    st.markdown('<div class="scan-title">Scan a Tomato Leaf</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="muted">Upload a single tomato leaf image to check for disease.</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Drag & drop your leaf image here or click Browse",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="visible",
    )

    st.caption("Supports JPG, PNG and WEBP • Best results: clear close-up leaf image")

    analyze = st.button("🔍 Get Diagnosis", type="primary", use_container_width=True)

    if uploaded is not None:
        preview = Image.open(uploaded).convert("RGB")
        st.image(preview, caption="Uploaded leaf image", use_container_width=True)

    st.markdown(
        '<div class="muted">🔒 Images are processed in memory and are not intentionally stored by this app.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------- METRICS ---------------------------

st.write("")
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        '<div class="metric"><div class="metric-value">9</div>'
        '<div class="metric-label">Disease Classes</div>'
        '<div class="metric-copy">Original model output categories</div></div>',
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        '<div class="metric"><div class="metric-value">224×224</div>'
        '<div class="metric-label">Input Size</div>'
        '<div class="metric-copy">Image size used for inference</div></div>',
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        '<div class="metric"><div class="metric-value">CLAHE</div>'
        '<div class="metric-label">Preprocessing</div>'
        '<div class="metric-copy">Contrast enhancement from training pipeline</div></div>',
        unsafe_allow_html=True,
    )

with m4:
    st.markdown(
        '<div class="metric"><div class="metric-value">DenseNet121</div>'
        '<div class="metric-label">Architecture</div>'
        '<div class="metric-copy">Trained TensorFlow/Keras classifier</div></div>',
        unsafe_allow_html=True,
    )

# ------------------------- RESULT ----------------------------

if analyze:
    if uploaded is None:
        st.warning("Please upload a tomato leaf image first.")
    else:
        try:
            image = Image.open(uploaded).convert("RGB")

            with st.spinner("Analyzing leaf image..."):
                predicted_key, confidence, probs, top_indices = run_prediction(image)

            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="result-name">{DISPLAY_NAMES[predicted_key]}</div>'
                f'<div class="result-conf">{confidence * 100:.1f}% — {confidence_level(confidence)}</div>'
                f'<div class="result-desc">{DISEASE_INFO[predicted_key]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("### Top predictions")

            for idx in top_indices:
                name = DISPLAY_NAMES[CLASS_NAMES[int(idx)]]
                score = float(probs[int(idx)])
                c1, c2 = st.columns([3.8, 1])
                with c1:
                    st.progress(score, text=name)
                with c2:
                    st.markdown(f"**{score * 100:.1f}%**")

        except Exception as exc:
            st.error("The image could not be analyzed.")
            st.code(str(exc))

# ---------------------- WHY TOMATOCARE -----------------------

st.markdown('<div id="about"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-kicker">Why TomatoCare AI?</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">From Leaf Image to Diagnosis</div>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;color:#aeb9c5;">A research prototype combining image preprocessing and deep learning classification.</p>',
    unsafe_allow_html=True,
)

f1, f2, f3, f4 = st.columns(4)

features = [
    ("🔬", "Detailed Symptom Analysis", "Analyzes leaf texture, color and visible patterns to detect disease signals."),
    ("✓", "Health Status Interpretation", "Classifies the uploaded leaf into one of the trained categories."),
    ("🧠", "Dense Feature Extraction", "Uses DenseNet121 feature representations for image classification."),
    ("🎯", "Targeted Disease Diagnosis", "Returns a ranked prediction list with confidence scores."),
]

for col, (icon, title, text) in zip((f1, f2, f3, f4), features):
    with col:
        st.markdown(
            f'<div class="feature"><div style="font-size:30px">{icon}</div>'
            f'<h3>{title}</h3><p>{text}</p></div>',
            unsafe_allow_html=True,
        )

# ---------------------- HOW IT WORKS -------------------------

st.markdown('<div id="how-it-works"></div>', unsafe_allow_html=True)
st.markdown("### How It Works")

steps = [
    ("01", "Upload", "Choose a clear tomato leaf image."),
    ("02", "Preprocess", "Resize to 224×224, apply CLAHE and normalize pixels."),
    ("03", "Predict", "DenseNet121 produces probabilities for 9 classes."),
    ("04", "Interpret", "The app shows the top result and ranked alternatives."),
]

cols = st.columns(4)
for col, (num, title, text) in zip(cols, steps):
    with col:
        st.markdown(
            f'<div class="feature"><div style="font-size:13px;color:#8fe0d2;font-weight:800;">STEP {num}</div>'
            f'<h3>{title}</h3><p>{text}</p></div>',
            unsafe_allow_html=True,
        )

# ---------------------- DATASET / TEAM / CONTACT ------------

st.markdown('<div id="dataset"></div>', unsafe_allow_html=True)
st.markdown("### Dataset & Model")
st.write(
    "The deployed model follows the uploaded training notebook's inference pipeline. "
    "The notebook used DenseNet121 with a 224×224 input and CLAHE preprocessing. "
    "The saved Keras model is loaded directly for inference."
)

st.markdown('<div id="team"></div>', unsafe_allow_html=True)
st.markdown("### Team")
st.write("TomatoCare AI • Final Year Project / Academic Research Prototype")

st.markdown('<div id="contact"></div>', unsafe_allow_html=True)
st.markdown("### Contact")
st.write("For project demonstration, academic feedback, or collaboration, contact the project team.")

st.markdown(
    """
<div class="footer">
    <b style="color:#dbe3ea;">TomatoCare AI</b><br>
    Final Year Project • Academic Research Prototype<br>
    Powered by DenseNet121 • TensorFlow/Keras • Streamlit
</div>
""",
    unsafe_allow_html=True,
)
