import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ============================================================
# TOMATO LEAF DISEASE DETECTOR
# Streamlit version - no Flask
# ============================================================

st.set_page_config(
    page_title="Tomato Leaf Disease Detector",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.keras")
IMG_SIZE = (224, 224)
MAX_FILE_SIZE = 5 * 1024 * 1024

# Exact class order from the original project
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

SCIENTIFIC_NAMES = {
    "Bacterial_spot": "Xanthomonas spp.",
    "Early_blight": "Alternaria solani",
    "Late_blight": "Phytophthora infestans",
    "Leaf_Miner": "Leaf Miner",
    "Leaf_Mold": "Passalora fulva",
    "Septoria_leaf_spot": "Septoria lycopersici",
    "Spider_mites": "Tetranychus spp.",
    "Verticillium_wilt": "Verticillium spp.",
    "healthy": "No disease detected",
}

INFO = {
    "Bacterial_spot": {
        "description": "A bacterial disease that can produce dark spots on tomato leaves.",
        "actions": ["Remove severely affected leaves.", "Improve spacing and air circulation.", "Avoid unnecessary overhead watering."],
    },
    "Early_blight": {
        "description": "A common fungal disease that can produce dark lesions and concentric ring patterns.",
        "actions": ["Remove and destroy infected leaves.", "Use locally recommended fungicide guidance.", "Ensure proper spacing and air circulation.", "Avoid prolonged leaf wetness.", "Rotate crops where appropriate."],
    },
    "Late_blight": {
        "description": "A serious disease that can spread rapidly under favorable environmental conditions.",
        "actions": ["Remove affected plant material where appropriate.", "Improve ventilation around plants.", "Monitor nearby plants regularly."],
    },
    "Leaf_Miner": {
        "description": "Leaf miner damage appears as winding trails or mines within leaf tissue.",
        "actions": ["Remove heavily affected leaves.", "Inspect the underside of leaves.", "Monitor plants regularly."],
    },
    "Leaf_Mold": {
        "description": "A fungal disease commonly associated with humid conditions.",
        "actions": ["Improve ventilation.", "Reduce prolonged leaf wetness.", "Remove severely affected foliage."],
    },
    "Septoria_leaf_spot": {
        "description": "A fungal disease that commonly produces numerous small lesions on tomato leaves.",
        "actions": ["Remove affected foliage.", "Improve plant spacing.", "Keep leaves as dry as practical."],
    },
    "Spider_mites": {
        "description": "Tiny pests that can cause stippling, discoloration and reduced plant vigor.",
        "actions": ["Inspect leaf undersides.", "Monitor plants frequently.", "Use an appropriate locally recommended management method."],
    },
    "Verticillium_wilt": {
        "description": "A soil-associated disease that can cause yellowing, wilting and reduced plant vigor.",
        "actions": ["Remove affected plants where appropriate.", "Use suitable crop rotation.", "Consider resistant varieties."],
    },
    "healthy": {
        "description": "The model classified this image as a healthy tomato leaf.",
        "actions": ["Continue regular crop monitoring.", "Maintain good airflow.", "Keep plants healthy with good crop management."],
    },
}

# ---------------- MODEL ----------------

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        from tensorflow.keras.models import load_model
        return load_model(MODEL_PATH, compile=False)
    except Exception:
        return None

model = get_model()

# ---------------- SESSION ----------------

defaults = {
    "page": "Dashboard",
    "image": None,
    "filename": "",
    "prediction": None,
    "history": [],
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------------- PREPROCESSING ----------------

def clahe_preprocessing(img):
    img = img.astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)

def preprocess(image):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(image).astype(np.float32)
    arr = clahe_preprocessing(arr)
    arr = arr / 255.0
    return np.expand_dims(arr, axis=0)

def predict(image):
    if model is None:
        return None

    x = preprocess(image)
    output = model.predict(x, verbose=0)

    if isinstance(output, (list, tuple)):
        output = output[0]

    probabilities = np.asarray(output).reshape(-1)

    if len(probabilities) != len(CLASS_NAMES):
        return None

    # Handles both softmax output and raw model scores safely.
    if not np.all(np.isfinite(probabilities)):
        return None

    if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0, atol=0.05):
        shifted = probabilities - np.max(probabilities)
        probabilities = np.exp(shifted)
        probabilities = probabilities / probabilities.sum()
    else:
        probabilities = probabilities / probabilities.sum()

    indexes = np.argsort(probabilities)[::-1][:3]

    result = []
    for index in indexes:
        cls = CLASS_NAMES[int(index)]
        result.append({
            "class": cls,
            "name": DISPLAY_NAMES[cls],
            "scientific": SCIENTIFIC_NAMES[cls],
            "confidence": float(probabilities[index] * 100),
        })

    return result

def do_prediction():
    if st.session_state.image is None:
        st.warning("Please choose a tomato leaf image first.")
        return

    if model is None:
        st.error("model.keras could not be loaded. Put model.keras in the same folder as app.py.")
        return

    with st.spinner("AI is analyzing the tomato leaf..."):
        result = predict(st.session_state.image)

    if result is None:
        st.error("Prediction failed. Please check that model.keras is the correct project model.")
        return

    st.session_state.prediction = result

    item = result[0]
    history_item = {
        "name": item["name"],
        "class": item["class"],
        "confidence": item["confidence"],
        "filename": st.session_state.filename,
    }

    # Avoid duplicate entry from reruns.
    if not any(
        h["filename"] == history_item["filename"]
        and h["class"] == history_item["class"]
        and round(h["confidence"], 2) == round(history_item["confidence"], 2)
        for h in st.session_state.history
    ):
        st.session_state.history.insert(0, history_item)
        st.session_state.history = st.session_state.history[:20]

# ---------------- CSS ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: #f6f9f7;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#003b28,#004c32,#002f20);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.brand {
    padding: 8px 5px 24px;
}

.logo {
    display:inline-flex;
    width:54px;
    height:54px;
    align-items:center;
    justify-content:center;
    border:2px solid #a9e84c;
    border-radius:16px;
    font-size:28px;
    margin-right:8px;
}

.brand-name {
    font-size:19px;
    font-weight:800;
}

.brand-sub {
    color:#b9e746 !important;
    font-size:11px;
    font-weight:700;
    margin-left:67px;
    margin-top:-13px;
}

.side-info {
    margin-top:25px;
    padding:17px;
    border-radius:16px;
    background:rgba(0,0,0,.18);
    border:1px solid rgba(170,235,100,.28);
    font-size:11px;
    line-height:1.7;
}

.page-title {
    font-size:34px;
    font-weight:800;
    color:#12231c;
    letter-spacing:-1.5px;
    margin-bottom:4px;
}

.page-subtitle {
    color:#718078;
    font-size:12px;
    margin-bottom:22px;
}

.card {
    background:#fff;
    border:1px solid #e2ebe5;
    border-radius:19px;
    padding:20px;
    box-shadow:0 10px 32px rgba(20,60,40,.06);
    margin-bottom:18px;
}

.card-title {
    font-size:15px;
    font-weight:800;
    color:#15251e;
}

.card-subtitle {
    color:#718078;
    font-size:11px;
    line-height:1.6;
    margin:7px 0 15px;
}

.upload-box {
    min-height:190px;
    border:2px dashed #58b875;
    border-radius:16px;
    background:#fbfefc;
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    text-align:center;
}

.upload-icon { font-size:43px; }
.upload-text { color:#52635a; font-size:11px; margin:8px; }

.result {
    padding:16px;
    border-radius:15px;
    background:#edf8ef;
    border:1px solid #d6ead9;
}

.result.danger {
    background:#fff2f3;
    border-color:#f0d5d8;
}

.result-name {
    font-size:26px;
    font-weight:800;
    color:#087f3f;
    margin:5px 0;
}

.danger .result-name { color:#bd3d45; }

.scientific {
    font-size:11px;
    color:#6c7d74;
}

.conf-label {
    margin-top:15px;
    color:#728179;
    font-size:10px;
}

.conf {
    color:#087f3f;
    font-size:25px;
    font-weight:800;
}

.info {
    margin-top:10px;
    background:#f7faf8;
    padding:12px;
    border-radius:12px;
    color:#52645b;
    font-size:10px;
    line-height:1.65;
}

.insight {
    min-height:125px;
    background:#fff;
    border:1px solid #e2ebe5;
    border-radius:16px;
    padding:17px;
    box-shadow:0 8px 24px rgba(20,60,40,.045);
}

.insight-icon { font-size:26px; }
.insight-title { font-size:12px; font-weight:800; margin-top:6px; }
.insight-text { color:#718078; font-size:10px; line-height:1.55; margin-top:6px; }

.action {
    color:#52645b;
    font-size:10px;
    line-height:1.6;
    margin:8px 0;
}

.check { color:#119246; font-weight:900; }

.stat {
    background:#fff;
    border:1px solid #e2ebe5;
    border-radius:15px;
    padding:15px;
}

.stat-number {
    color:#087f3f;
    font-size:24px;
    font-weight:800;
}

.stat-label {
    color:#718078;
    font-size:9px;
    margin-top:3px;
}

.footer {
    background:linear-gradient(90deg,#edf8ef,#f8fcf8);
    border:1px solid #dcebdd;
    border-radius:14px;
    padding:14px;
    text-align:center;
    color:#52645b;
    font-size:10px;
    line-height:1.7;
    margin-top:20px;
}

.stButton > button {
    border-radius:10px !important;
    min-height:41px !important;
    font-weight:700 !important;
}

button[kind="primary"] {
    background:linear-gradient(135deg,#099348,#06753a) !important;
    border:0 !important;
}

[data-testid="stFileUploaderDropzone"] {
    background:transparent !important;
    border:0 !important;
}

.history-row {
    padding:11px 0;
    border-bottom:1px solid #edf1ee;
}

.history-name {
    font-size:10px;
    font-weight:800;
}

.history-meta {
    color:#77867f;
    font-size:9px;
    margin-top:3px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div class="brand">
        <span class="logo">🌿</span>
        <span class="brand-name">Tomato Leaf</span>
        <div class="brand-sub">Disease Detector</div>
    </div>
    """, unsafe_allow_html=True)

    pages = [
        "🏠  Dashboard",
        "🔬  Analyze Leaf",
        "📖  Disease Guide",
        "🛡️  Prevention Tips",
        "◷  History",
        "📊  Statistics",
        "ⓘ  About Project",
    ]

    page_names = [
        "Dashboard",
        "Analyze Leaf",
        "Disease Guide",
        "Prevention Tips",
        "History",
        "Statistics",
        "About Project",
    ]

    active_index = page_names.index(st.session_state.page)

    selected = st.radio(
        "Menu",
        pages,
        index=active_index,
        label_visibility="collapsed",
    )

    selected_name = selected.split("  ", 1)[1]

    if selected_name != st.session_state.page:
        st.session_state.page = selected_name
        st.rerun()

    st.markdown("""
    <div class="side-info">
        <b>⚡ AI Powered Detection</b>
        <br><br>
        Advanced deep learning model for accurate and reliable tomato leaf disease classification.
        <br><br>
        🌱 🍅 🌿 🍅 🌱
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# DASHBOARD
# ============================================================

if st.session_state.page == "Dashboard":

    st.markdown("""
    <div class="page-title">AI-Powered Tomato Leaf Analysis 🌿</div>
    <div class="page-subtitle">Upload a leaf image and identify potential diseases with the power of AI.</div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1.18], gap="large")

    with left:
        st.markdown("""
        <div class="card">
        <div class="card-title">☁️ Upload Tomato Leaf Image</div>
        <div class="card-subtitle">Upload a clear image of the tomato leaf to detect possible diseases.</div>
        <div class="upload-box">
            <div class="upload-icon">☁️</div>
            <div class="upload-text">Drag & drop your leaf image here<br>or choose an image below</div>
        </div>
        """, unsafe_allow_html=True)

        file = st.file_uploader(
            "Choose Image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="main_upload",
        )

        if file is not None:
            if file.size > MAX_FILE_SIZE:
                st.error("Maximum image size is 5 MB.")
            else:
                st.session_state.image = Image.open(file).convert("RGB")
                st.session_state.filename = file.name
                st.success("✓ Image selected successfully")

        st.markdown("""
        <div style="text-align:center;color:#718078;font-size:9px;margin:8px 0;">
        Supported formats: JPG, JPEG, PNG • Max size: 5MB
        </div>
        """, unsafe_allow_html=True)

        if st.button("✨  Submit & Analyze", type="primary", use_container_width=True):
            do_prediction()
            st.rerun()

        st.markdown("""
        <div class="info">💡 <b>Tip:</b> Use a clear image in good lighting for best results.</div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="card">
        <div class="card-title">📈 Prediction Result</div>
        <div class="card-subtitle">AI-assisted classification result from your trained model.</div>
        """, unsafe_allow_html=True)

        if st.session_state.prediction is None:
            st.markdown("""
            <div style="height:290px;display:flex;align-items:center;justify-content:center;text-align:center;color:#8a9891;font-size:11px;">
                Upload a tomato leaf image and click<br>
                <b>Submit & Analyze</b> to see the result.
            </div>
            """, unsafe_allow_html=True)
        else:
            best = st.session_state.prediction[0]
            healthy = best["class"] == "healthy"

            img_col, detail_col = st.columns([1.02, .98], gap="medium")

            with img_col:
                st.image(st.session_state.image, use_container_width=True)

            with detail_col:
                st.markdown(f"""
                <div class="result {' ' if healthy else 'danger'}">
                    <div style="font-size:10px;font-weight:800;color:#6b7b73;">
                    ● {'Healthy Leaf' if healthy else 'Disease Detected'}
                    </div>
                    <div class="result-name">{best["name"]}</div>
                    <div class="scientific">({best["scientific"]})</div>
                    <div class="conf-label">Confidence Score</div>
                    <div class="conf">{best["confidence"]:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

                st.progress(min(best["confidence"] / 100, 1.0))

                st.markdown(
                    f'<div class="info">🛡️ {INFO[best["class"]]["description"]}</div>',
                    unsafe_allow_html=True,
                )

            a, b = st.columns(2)

            with a:
                if st.button("📄  View Details", use_container_width=True):
                    st.session_state.page = "Analyze Leaf"
                    st.rerun()

            with b:
                if st.button("⟳  Analyze Another", use_container_width=True):
                    st.session_state.image = None
                    st.session_state.filename = ""
                    st.session_state.prediction = None
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Why AI-assisted detection?")

    benefits = [
        ("🛡️", "Early Detection", "Detect possible conditions early and support timely crop monitoring."),
        ("🎯", "Better Yield", "Healthy plants can support better crop performance and productivity."),
        ("🌱", "Save Resources", "Use AI-assisted information to support informed crop decisions."),
        ("📊", "Smart Farming", "Bring image classification technology into modern agriculture."),
    ]

    cols = st.columns(4)

    for col, (icon, title, text) in zip(cols, benefits):
        with col:
            st.markdown(f"""
            <div class="insight">
                <div class="insight-icon">{icon}</div>
                <div class="insight-title">{title}</div>
                <div class="insight-text">{text}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### Analysis Overview")

    c1, c2, c3 = st.columns([1.1, 1, 1])

    active = st.session_state.prediction[0]["class"] if st.session_state.prediction else "Early_blight"

    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">ⓘ About {DISPLAY_NAMES[active]}</div>
            <div class="info" style="margin-top:12px;">{INFO[active]["description"]}</div>
            <div style="margin-top:12px;">
                <div class="action"><span class="check">✓</span> Regularly inspect leaves.</div>
                <div class="action"><span class="check">✓</span> Monitor environmental conditions.</div>
                <div class="action"><span class="check">✓</span> Act early when symptoms appear.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        actions = INFO[active]["actions"]
        action_html = "".join(
            f'<div class="action"><span class="check">✓</span> {x}</div>'
            for x in actions[:5]
        )

        st.markdown(f"""
        <div class="card">
            <div class="card-title">🛡️ Recommended Actions</div>
            <div style="margin-top:12px;">{action_html}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="card"><div class="card-title">◷ Recent History</div>', unsafe_allow_html=True)

        if st.session_state.history:
            for item in st.session_state.history[:4]:
                st.markdown(f"""
                <div class="history-row">
                    <div class="history-name">{item["name"]}
                    <span style="float:right;color:#087f3f;">{'Healthy' if item["class"] == "healthy" else 'Detected'}</span></div>
                    <div class="history-meta">{item["confidence"]:.2f}% • {item["filename"]}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="info">No scans yet. Upload your first leaf image.</div>', unsafe_allow_html=True)

        if st.button("▣  View All History", use_container_width=True):
            st.session_state.page = "History"
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    total = len(st.session_state.history)
    diseases = sum(x["class"] != "healthy" for x in st.session_state.history)
    healthy_count = sum(x["class"] == "healthy" for x in st.session_state.history)
    avg = np.mean([x["confidence"] for x in st.session_state.history]) if total else 0

    st.markdown("### 📊 Statistics Overview")

    for col, number, label in zip(
        st.columns(4),
        [total, diseases, healthy_count, f"{avg:.2f}%"],
        ["Total Predictions", "Diseases Detected", "Healthy Leaves", "Average Confidence"],
    ):
        with col:
            st.markdown(f"""
            <div class="stat">
                <div class="stat-number">{number}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
    🌱 Protect your crops, increase your yield, and build a better tomorrow with AI.
    <br><b>Smart Farming, Better Future. 🌿</b>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# ANALYZE LEAF
# ============================================================

elif st.session_state.page == "Analyze Leaf":

    st.markdown("""
    <div class="page-title">Analyze Tomato Leaf 🔬</div>
    <div class="page-subtitle">Inspect the uploaded image and complete AI prediction.</div>
    """, unsafe_allow_html=True)

    if st.session_state.image is None:
        st.markdown("""
        <div class="card" style="text-align:center;padding:55px;">
            <div style="font-size:42px;">🍃</div>
            <div class="card-title">No image selected</div>
            <div class="card-subtitle">Go to Dashboard and upload a tomato leaf image.</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("← Go to Dashboard", type="primary"):
            st.session_state.page = "Dashboard"
            st.rerun()

    else:
        left, right = st.columns([1, 1.05], gap="large")

        with left:
            st.markdown('<div class="card"><div class="card-title">Uploaded Leaf</div>', unsafe_allow_html=True)
            st.image(st.session_state.image, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="card"><div class="card-title">Complete AI Analysis</div>', unsafe_allow_html=True)

            if st.session_state.prediction is None:
                if st.button("✨  Run AI Analysis", type="primary", use_container_width=True):
                    do_prediction()
                    st.rerun()
            else:
                best = st.session_state.prediction[0]

                st.markdown(f"""
                <div class="result {' ' if best["class"] == "healthy" else "danger"}">
                    <div style="font-size:10px;font-weight:800;color:#6b7b73;">
                    ● {'Healthy Leaf' if best["class"] == "healthy" else 'Disease Detected'}
                    </div>
                    <div class="result-name">{best["name"]}</div>
                    <div class="scientific">({best["scientific"]})</div>
                    <div class="conf-label">Confidence Score</div>
                    <div class="conf">{best["confidence"]:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

                st.progress(min(best["confidence"] / 100, 1.0))

                st.markdown("#### Top-3 Predictions")

                for i, item in enumerate(st.session_state.prediction, 1):
                    st.markdown(f"""
                    <div class="history-row">
                        <span class="history-name">{i}. {item["name"]}</span>
                        <span style="float:right;color:#087f3f;font-size:10px;font-weight:800;">
                        {item["confidence"]:.2f}%
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("#### Recommended Actions")

                for action in INFO[best["class"]]["actions"]:
                    st.markdown(
                        f'<div class="action"><span class="check">✓</span> {action}</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("</div>", unsafe_allow_html=True)

        a, b = st.columns(2)

        with a:
            if st.button("← Back to Dashboard", use_container_width=True):
                st.session_state.page = "Dashboard"
                st.rerun()

        with b:
            if st.button("⟳ Analyze Another Leaf", use_container_width=True):
                st.session_state.image = None
                st.session_state.filename = ""
                st.session_state.prediction = None
                st.rerun()

# ============================================================
# DISEASE GUIDE
# ============================================================

elif st.session_state.page == "Disease Guide":

    st.markdown("""
    <div class="page-title">Disease Guide 📖</div>
    <div class="page-subtitle">All disease classes supported by the project model.</div>
    """, unsafe_allow_html=True)

    cols = st.columns(3)

    for i, cls in enumerate(CLASS_NAMES):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="card">
                <div class="card-title">{DISPLAY_NAMES[cls]}</div>
                <div class="scientific" style="margin-top:4px;">{SCIENTIFIC_NAMES[cls]}</div>
                <div class="info" style="margin-top:12px;">{INFO[cls]["description"]}</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# PREVENTION
# ============================================================

elif st.session_state.page == "Prevention Tips":

    st.markdown("""
    <div class="page-title">Prevention Tips 🛡️</div>
    <div class="page-subtitle">Practical practices for healthier tomato plants.</div>
    """, unsafe_allow_html=True)

    tips = [
        ("🌬️", "Air Circulation", "Maintain suitable spacing between plants."),
        ("💧", "Water Management", "Avoid unnecessary overhead watering and prolonged leaf wetness."),
        ("🔍", "Regular Monitoring", "Inspect both sides of leaves regularly."),
        ("🌱", "Plant Hygiene", "Remove severely affected plant material where appropriate."),
        ("🔄", "Crop Rotation", "Use suitable rotation practices."),
        ("🧤", "Tool Hygiene", "Keep tools and growing areas clean."),
    ]

    cols = st.columns(3)

    for i, (icon, title, text) in enumerate(tips):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="card">
                <div style="font-size:30px;">{icon}</div>
                <div class="card-title">{title}</div>
                <div class="info" style="margin-top:10px;">{text}</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# HISTORY
# ============================================================

elif st.session_state.page == "History":

    st.markdown("""
    <div class="page-title">Scan History ◷</div>
    <div class="page-subtitle">Predictions made during this session.</div>
    """, unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No scan history yet.")
    else:
        for item in st.session_state.history:
            st.markdown(f"""
            <div class="card" style="padding:15px;">
                <div class="card-title">{item["name"]}</div>
                <div class="history-meta">
                    {item["confidence"]:.2f}% confidence • {item["filename"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("Clear Session History"):
            st.session_state.history = []
            st.rerun()

# ============================================================
# STATISTICS
# ============================================================

elif st.session_state.page == "Statistics":

    st.markdown("""
    <div class="page-title">Statistics 📊</div>
    <div class="page-subtitle">Overview of AI predictions in this session.</div>
    """, unsafe_allow_html=True)

    total = len(st.session_state.history)
    disease_count = sum(x["class"] != "healthy" for x in st.session_state.history)
    healthy_count = sum(x["class"] == "healthy" for x in st.session_state.history)
    avg = np.mean([x["confidence"] for x in st.session_state.history]) if total else 0

    cols = st.columns(4)

    for col, number, label in zip(
        cols,
        [total, disease_count, healthy_count, f"{avg:.2f}%"],
        ["Total Predictions", "Diseases Detected", "Healthy Leaves", "Average Confidence"],
    ):
        with col:
            st.markdown(f"""
            <div class="stat">
                <div class="stat-number">{number}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    if st.session_state.history:
        counts = {}
        for item in st.session_state.history:
            counts[item["name"]] = counts.get(item["name"], 0) + 1
        st.markdown("### Prediction Distribution")
        st.bar_chart(counts)

# ============================================================
# ABOUT
# ============================================================

elif st.session_state.page == "About Project":

    st.markdown("""
    <div class="page-title">About Project ⓘ</div>
    <div class="page-subtitle">Tomato Leaf Disease Detection using Deep Learning.</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="card-title">🍅 Tomato Leaf Disease Detector</div>
        <div class="info" style="margin-top:12px;">
        This application provides a professional web interface for the project's trained tomato leaf disease classification model.
        <br><br>
        The original project uses 224 × 224 RGB input with CLAHE preprocessing followed by /255 normalization.
        </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)

    for col, number, label in zip(
        cols,
        ["224×224", "9", "DenseNet121", "Softmax"],
        ["Input Size", "Classes", "Architecture", "Output"],
    ):
        with col:
            st.markdown(f"""
            <div class="stat">
                <div class="stat-number">{number}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
    <b>Smart Farming, Better Future. 🌿</b>
    </div>
    """, unsafe_allow_html=True)
