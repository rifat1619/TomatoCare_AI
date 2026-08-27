import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# ============================================================
# TOMATO LEAF DISEASE DETECTOR - STREAMLIT APP
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

# Exact 9-class order used by the project
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
    "Bacterial_spot": {
        "description": "A bacterial disease that can produce dark spots on tomato leaves.",
        "actions": [
            "Remove severely affected leaves.",
            "Improve plant spacing and airflow.",
            "Avoid unnecessary overhead watering.",
        ],
    },
    "Early_blight": {
        "description": "A common fungal disease associated with dark lesions and concentric patterns on tomato leaves.",
        "actions": [
            "Remove severely affected leaves.",
            "Improve spacing and air circulation.",
            "Avoid prolonged leaf wetness.",
            "Follow locally recommended disease-management practices.",
        ],
    },
    "Late_blight": {
        "description": "A serious tomato disease that can spread rapidly under favorable environmental conditions.",
        "actions": [
            "Remove affected plant material.",
            "Improve ventilation around plants.",
            "Monitor nearby plants regularly.",
            "Follow locally appropriate management guidance.",
        ],
    },
    "Leaf_Miner": {
        "description": "Leaf miner damage commonly appears as winding trails or mines within leaf tissue.",
        "actions": [
            "Remove heavily affected leaves.",
            "Inspect leaves regularly.",
            "Monitor plants for further activity.",
        ],
    },
    "Leaf_Mold": {
        "description": "Leaf mold is a fungal disease commonly associated with humid conditions.",
        "actions": [
            "Improve ventilation.",
            "Reduce prolonged leaf wetness.",
            "Remove severely affected foliage.",
        ],
    },
    "Septoria_leaf_spot": {
        "description": "A fungal leaf-spot disease that commonly produces numerous small lesions.",
        "actions": [
            "Remove affected foliage.",
            "Improve plant spacing.",
            "Keep leaves as dry as practical.",
            "Monitor plants regularly.",
        ],
    },
    "Spider_mites": {
        "description": "Spider mites are tiny pests that can cause stippling and discoloration of tomato leaves.",
        "actions": [
            "Inspect the underside of leaves.",
            "Monitor plants frequently.",
            "Use an appropriate locally recommended management method.",
        ],
    },
    "Verticillium_wilt": {
        "description": "A soil-borne disease that can cause yellowing, wilting and reduced plant vigor.",
        "actions": [
            "Remove affected plants where appropriate.",
            "Use suitable crop rotation.",
            "Consider resistant varieties.",
        ],
    },
    "healthy": {
        "description": "The model classified this image as a healthy tomato leaf.",
        "actions": [
            "Continue regular crop monitoring.",
            "Maintain good airflow.",
            "Continue good crop-management practices.",
        ],
    },
}

# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        st.error("model.keras not found. Put model.keras beside app.py.")
        st.stop()
    return load_model(MODEL_PATH, compile=False)


model = get_model()


# ============================================================
# PREPROCESSING
# ============================================================

def clahe_preprocessing(img):
    img = img.astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return img.astype(np.float32)


def preprocess(image):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(image).astype(np.float32)
    arr = clahe_preprocessing(arr)
    arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


def predict(image):
    x = preprocess(image)
    probabilities = model.predict(x, verbose=0)[0]

    indices = np.argsort(probabilities)[::-1][:3]

    return [
        {
            "class": CLASS_NAMES[i],
            "name": DISPLAY_NAMES[CLASS_NAMES[i]],
            "confidence": float(probabilities[i] * 100),
        }
        for i in indices
    ]


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_file" not in st.session_state:
    st.session_state.last_file = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ============================================================
# CSS
# IMPORTANT: HTML is deliberately kept flush-left inside
# markdown strings so Streamlit never treats it as code.
# ============================================================

st.markdown(
"""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}

.stApp {
    background: #f5f8f6;
}

[data-testid="stHeader"] {
    background: rgba(245,248,246,0.92);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #052d1f 0%, #063e28 55%, #052d1f 100%);
}

[data-testid="stSidebar"] * {
    color: white;
}

.brand-wrap {
    padding: 8px 3px 24px 3px;
}

.brand-logo {
    display: inline-flex;
    width: 54px;
    height: 54px;
    align-items: center;
    justify-content: center;
    border: 2px solid #a5dd4d;
    border-radius: 16px;
    font-size: 27px;
    vertical-align: middle;
    margin-right: 9px;
}

.brand-name {
    display: inline-block;
    vertical-align: middle;
    font-size: 18px;
    font-weight: 800;
}

.brand-sub {
    margin-left: 67px;
    margin-top: -13px;
    color: #b6dd39 !important;
    font-size: 12px;
    font-weight: 700;
}

.side-panel {
    margin-top: 28px;
    padding: 18px;
    border-radius: 17px;
    border: 1px solid rgba(150,230,170,.22);
    background: rgba(255,255,255,.06);
    font-size: 11px;
    line-height: 1.7;
}

.main-heading {
    color: #13251e;
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin: 3px 0 4px 0;
}

.main-subheading {
    color: #708078;
    font-size: 13px;
    margin-bottom: 24px;
}

.card {
    background: #ffffff;
    border: 1px solid #e3ebe6;
    border-radius: 20px;
    padding: 21px;
    box-shadow: 0 14px 40px rgba(15,53,35,.07);
    margin-bottom: 18px;
}

.card-title {
    color: #15251e;
    font-size: 16px;
    font-weight: 800;
}

.card-subtitle {
    color: #708078;
    font-size: 11px;
    line-height: 1.6;
    margin-top: 6px;
    margin-bottom: 16px;
}

.tip {
    background: #eef8ef;
    border: 1px solid #dcecdc;
    border-radius: 12px;
    padding: 12px;
    color: #52655a;
    font-size: 10px;
    line-height: 1.6;
    margin-top: 12px;
}

.prediction {
    background: #eef8ef;
    border: 1px solid #d9ecdc;
    border-radius: 15px;
    padding: 15px;
}

.prediction.disease {
    background: #fff1f2;
    border-color: #f1d7d9;
}

.prediction-status {
    color: #65766d;
    font-size: 10px;
    font-weight: 800;
}

.prediction-name {
    color: #0b5a36;
    font-size: 27px;
    font-weight: 800;
    letter-spacing: -.6px;
    margin: 5px 0;
}

.disease .prediction-name {
    color: #bd3f47;
}

.confidence-label {
    color: #708078;
    font-size: 10px;
    margin-top: 14px;
}

.confidence-number {
    color: #168447;
    font-size: 24px;
    font-weight: 800;
    margin-top: 2px;
}

.info-box {
    background: #f7faf8;
    border-radius: 12px;
    padding: 12px;
    color: #52645a;
    font-size: 10px;
    line-height: 1.65;
    margin-top: 13px;
}

.action {
    color: #52645a;
    font-size: 10px;
    line-height: 1.55;
    margin: 8px 0;
}

.check {
    color: #1b9c4e;
    font-weight: 900;
}

.insight {
    background: #ffffff;
    border: 1px solid #e3ebe6;
    border-radius: 17px;
    padding: 19px;
    min-height: 145px;
    box-shadow: 0 10px 28px rgba(15,53,35,.05);
}

.insight-icon {
    font-size: 26px;
    margin-bottom: 9px;
}

.insight-title {
    color: #15251e;
    font-size: 12px;
    font-weight: 800;
}

.insight-text {
    color: #708078;
    font-size: 10px;
    line-height: 1.6;
    margin-top: 6px;
}

.stat {
    background: #ffffff;
    border: 1px solid #e3ebe6;
    border-radius: 16px;
    padding: 17px;
    box-shadow: 0 10px 28px rgba(15,53,35,.05);
}

.stat-number {
    color: #168447;
    font-size: 26px;
    font-weight: 800;
}

.stat-label {
    color: #708078;
    font-size: 10px;
    margin-top: 3px;
}

.footer {
    margin-top: 20px;
    padding: 16px;
    text-align: center;
    border: 1px solid #dcebdc;
    border-radius: 15px;
    background: linear-gradient(90deg,#edf8ef,#f9fcf8);
    color: #596a61;
    font-size: 10px;
    line-height: 1.7;
}

[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #63b878 !important;
    border-radius: 17px !important;
    background: linear-gradient(180deg,#fbfefc,#f3faf5) !important;
    min-height: 215px;
}

[data-testid="stFileUploaderDropzone"] button {
    background: #168447 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 9px !important;
}

[data-testid="stImage"] img {
    border-radius: 15px;
}

.stButton > button {
    border-radius: 10px;
    font-weight: 700;
}

@media (max-width: 900px) {
    .main-heading {
        font-size: 27px;
    }
}
</style>""",
unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
"""<div class="brand-wrap">
<span class="brand-logo">🌿</span>
<span class="brand-name">Tomato Leaf</span>
<div class="brand-sub">Disease Detector</div>
</div>""",
        unsafe_allow_html=True,
    )

    menu = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Analyze Leaf",
            "Disease Guide",
            "Prevention Tips",
            "History",
            "Statistics",
            "About Project",
        ],
        label_visibility="collapsed",
    )

    st.markdown(
"""<div class="side-panel">
<b>⚡ AI Powered Detection</b>
<br><br>
Advanced deep learning model for accurate and reliable tomato leaf disease detection.
<br><br>
🍅 🌿 🍅 🌿 🍅
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
"""<div style="margin-top:35px;font-size:10px;opacity:.55;">
© 2026 Tomato Leaf Disease Detector
<br>
All rights reserved.
</div>""",
        unsafe_allow_html=True,
    )


# ============================================================
# PAGE: DISEASE GUIDE
# ============================================================

if menu == "Disease Guide":
    st.markdown(
"""<div class="main-heading">Disease Guide</div>
<div class="main-subheading">Conditions supported by the trained AI model.</div>""",
        unsafe_allow_html=True,
    )

    cols = st.columns(3)

    for i, class_name in enumerate(CLASS_NAMES):
        info = DISEASE_INFO[class_name]

        with cols[i % 3]:
            actions_html = "".join(
                f'<div class="action"><span class="check">✓</span> {action}</div>'
                for action in info["actions"]
            )

            st.markdown(
f"""<div class="card">
<div class="card-title">{DISPLAY_NAMES[class_name]}</div>
<div class="info-box">{info["description"]}</div>
<div style="margin-top:13px;font-size:10px;font-weight:800;color:#31443a;">Recommended Actions</div>
{actions_html}
</div>""",
                unsafe_allow_html=True,
            )

    st.stop()


# ============================================================
# PAGE: PREVENTION
# ============================================================

if menu == "Prevention Tips":
    st.markdown(
"""<div class="main-heading">Prevention Tips</div>
<div class="main-subheading">General practices for healthier tomato plants.</div>""",
        unsafe_allow_html=True,
    )

    tips = [
        ("🌬️", "Improve Air Circulation", "Keep suitable spacing between plants to reduce prolonged humidity."),
        ("💧", "Manage Leaf Wetness", "Avoid unnecessary overhead watering and prolonged leaf wetness."),
        ("🔍", "Monitor Regularly", "Inspect tomato leaves frequently for unusual symptoms."),
        ("🌱", "Remove Affected Leaves", "Remove severely affected plant material where appropriate."),
        ("🔄", "Crop Rotation", "Use suitable crop rotation practices for better crop management."),
        ("🧤", "Maintain Hygiene", "Keep tools and growing areas clean and remove infected debris appropriately."),
    ]

    cols = st.columns(3)

    for i, (icon, title, description) in enumerate(tips):
        with cols[i % 3]:
            st.markdown(
f"""<div class="card">
<div style="font-size:27px;">{icon}</div>
<div class="card-title">{title}</div>
<div class="info-box">{description}</div>
</div>""",
                unsafe_allow_html=True,
            )

    st.stop()


# ============================================================
# PAGE: ABOUT
# ============================================================

if menu == "About Project":
    st.markdown(
"""<div class="main-heading">About the Project</div>
<div class="main-subheading">AI-powered tomato leaf disease classification system.</div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
"""<div class="card">
<div class="card-title">🍅 Tomato Leaf Disease Detector</div>
<div class="info-box">
This application provides a web interface for the trained deep-learning model used in the project.
<br><br>
The project uses a DenseNet121-based architecture with global average pooling, global max pooling, concatenation, batch normalization, dense layers and a softmax classification output.
</div>
</div>""",
        unsafe_allow_html=True,
    )

    cols = st.columns(4)

    for col, number, label in zip(
        cols,
        ["224×224", "9", "DenseNet121", "Softmax"],
        ["Input Size", "Classes", "Architecture", "Output"],
    ):
        with col:
            st.markdown(
f"""<div class="stat">
<div class="stat-number">{number}</div>
<div class="stat-label">{label}</div>
</div>""",
                unsafe_allow_html=True,
            )

    st.stop()


# ============================================================
# PAGE: HISTORY
# ============================================================

if menu == "History":
    st.markdown(
"""<div class="main-heading">Recent History</div>
<div class="main-subheading">Predictions made during this app session.</div>""",
        unsafe_allow_html=True,
    )

    if not st.session_state.history:
        st.info("No predictions yet. Upload your first leaf image.")
    else:
        for item in st.session_state.history:
            st.markdown(
f"""<div class="card">
<div class="card-title">{item["name"]}</div>
<div class="info-box">
Confidence: <b>{item["confidence"]:.2f}%</b>
<br>
File: {item["filename"]}
</div>
</div>""",
                unsafe_allow_html=True,
            )

    st.stop()


# ============================================================
# PAGE: STATISTICS
# ============================================================

if menu == "Statistics":
    st.markdown(
"""<div class="main-heading">Statistics Overview</div>
<div class="main-subheading">Overview of predictions made during this session.</div>""",
        unsafe_allow_html=True,
    )

    history = st.session_state.history

    total = len(history)
    disease_count = sum(x["class"] != "healthy" for x in history)
    healthy_count = sum(x["class"] == "healthy" for x in history)
    average = (
        float(np.mean([x["confidence"] for x in history]))
        if history else 0
    )

    cols = st.columns(4)

    stats = [
        (total, "Total Predictions"),
        (disease_count, "Diseases Detected"),
        (healthy_count, "Healthy Leaves"),
        (f"{average:.2f}%", "Average Confidence"),
    ]

    for col, (number, label) in zip(cols, stats):
        with col:
            st.markdown(
f"""<div class="stat">
<div class="stat-number">{number}</div>
<div class="stat-label">{label}</div>
</div>""",
                unsafe_allow_html=True,
            )

    st.markdown("### Prediction Distribution")

    if history:
        counts = {}
        for item in history:
            counts[item["name"]] = counts.get(item["name"], 0) + 1
        st.bar_chart(counts)
    else:
        st.info("No predictions available yet.")

    st.stop()


# ============================================================
# MAIN DASHBOARD
# ============================================================

st.markdown(
"""<div class="main-heading">AI-Powered Tomato Leaf Analysis 🌿</div>
<div class="main-subheading">Upload a leaf image and identify potential diseases with AI.</div>""",
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1.15], gap="large")


# ============================================================
# UPLOAD CARD
# ============================================================

with left:
    st.markdown(
"""<div class="card">
<div class="card-title">⇧ Upload Tomato Leaf Image</div>
<div class="card-subtitle">
Upload a clear image of the tomato leaf to detect possible diseases.
</div>""",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Drag & drop your leaf image here",
        type=["jpg", "jpeg", "png"],
        help="Maximum file size: 5 MB",
    )

    st.markdown(
"""<div style="color:#708078;font-size:10px;margin-top:8px;">
Supported formats: JPG, JPEG, PNG &nbsp; • &nbsp; Max size: 5MB
</div>
<div class="tip">
💡 <b>Tip:</b> Use a clear image in good lighting for best results.
</div>
</div>""",
        unsafe_allow_html=True,
    )


# ============================================================
# PREDICTION CARD
# ============================================================

current_top3 = None
current_image = None

with right:
    st.markdown(
"""<div class="card">
<div class="card-title">📈 Prediction Result</div>
<div class="card-subtitle">
AI-assisted classification result from your trained model.
</div>""",
        unsafe_allow_html=True,
    )

    if uploaded_file is None:
        st.markdown(
"""<div style="min-height:300px;display:flex;align-items:center;justify-content:center;text-align:center;color:#87958d;font-size:12px;">
Upload an image to see the prediction result.
</div>""",
            unsafe_allow_html=True,
        )

    else:
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error("Image size must be less than 5 MB.")
        else:
            try:
                current_image = Image.open(uploaded_file).convert("RGB")

                with st.spinner("Analyzing leaf..."):
                    current_top3 = predict(current_image)

                best = current_top3[0]
                is_healthy = best["class"] == "healthy"

                image_col, result_col = st.columns([1.05, 0.95])

                with image_col:
                    st.image(
                        current_image,
                        caption=uploaded_file.name,
                        use_container_width=True,
                    )

                with result_col:
                    prediction_class = "" if is_healthy else "disease"
                    status = "Healthy Leaf" if is_healthy else "Disease Detected"

                    st.markdown(
f"""<div class="prediction {prediction_class}">
<div class="prediction-status">● {status}</div>
<div class="prediction-name">{best["name"]}</div>
<div class="mini-muted">{best["class"]}</div>
<div class="confidence-label">Confidence Score</div>
<div class="confidence-number">{best["confidence"]:.2f}%</div>
</div>""",
                        unsafe_allow_html=True,
                    )

                    st.progress(
                        min(best["confidence"] / 100, 1.0)
                    )

                info = DISEASE_INFO[best["class"]]

                st.markdown(
f"""<div class="info-box">
<b>About {best["name"]}</b>
<br><br>
{info["description"]}
</div>""",
                    unsafe_allow_html=True,
                )

                st.markdown("#### Recommended Actions")

                for action in info["actions"]:
                    st.markdown(
f"""<div class="action"><span class="check">✓</span> {action}</div>""",
                        unsafe_allow_html=True,
                    )

                st.markdown("#### Top-3 Predictions")

                for rank, item in enumerate(current_top3, 1):
                    st.markdown(
f"""<div style="padding:9px 0;border-bottom:1px solid #edf1ee;font-size:11px;">
<b>{rank}. {item["name"]}</b>
<span style="float:right;color:#168447;font-weight:800;">
{item["confidence"]:.2f}%
</span>
</div>""",
                        unsafe_allow_html=True,
                    )

                # Store history only once for this uploaded file.
                signature = (
                    uploaded_file.name,
                    uploaded_file.size,
                    round(best["confidence"], 4),
                )

                if st.session_state.last_file != signature:
                    st.session_state.history.insert(
                        0,
                        {
                            "name": best["name"],
                            "class": best["class"],
                            "confidence": best["confidence"],
                            "filename": uploaded_file.name,
                        },
                    )
                    st.session_state.history = st.session_state.history[:10]
                    st.session_state.last_file = signature

                st.session_state.last_result = best["class"]

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# INSIGHTS
# ============================================================

st.markdown("### Why AI-assisted detection?")

insights = [
    ("🛡️", "Early Detection", "Detect possible leaf conditions early and support timely crop monitoring."),
    ("🎯", "Better Yield", "Healthy plants can support better crop performance and productivity."),
    ("🌱", "Save Resources", "Use AI-assisted information to support more informed crop decisions."),
    ("📊", "Smart Farming", "Bring image classification technology into agricultural workflows."),
]

cols = st.columns(4)

for col, (icon, title, text) in zip(cols, insights):
    with col:
        st.markdown(
f"""<div class="insight">
<div class="insight-icon">{icon}</div>
<div class="insight-title">{title}</div>
<div class="insight-text">{text}</div>
</div>""",
            unsafe_allow_html=True,
        )


# ============================================================
# BOTTOM THREE CARDS
# ============================================================

st.markdown("### Analysis Overview")

b1, b2, b3 = st.columns([1.1, 1, 1])

# About disease
with b1:
    if current_top3:
        bottom_class = current_top3[0]["class"]
    elif st.session_state.last_result:
        bottom_class = st.session_state.last_result
    else:
        bottom_class = "Early_blight"

    info = DISEASE_INFO[bottom_class]

    st.markdown(
f"""<div class="card">
<div class="card-title">ⓘ About {DISPLAY_NAMES[bottom_class]}</div>
<div class="info-box">{info["description"]}</div>
<div style="margin-top:13px;">
<div class="action"><span class="check">✓</span> AI-assisted image classification</div>
<div class="action"><span class="check">✓</span> 224 × 224 model input</div>
<div class="action"><span class="check">✓</span> Confidence-based prediction</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )


# Recommended actions
with b2:
    st.markdown(
"""<div class="card">
<div class="card-title">🛡️ Recommended Actions</div>
<div style="margin-top:13px;">
<div class="action"><span class="check">✓</span> Remove severely affected leaves</div>
<div class="action"><span class="check">✓</span> Ensure proper spacing and airflow</div>
<div class="action"><span class="check">✓</span> Avoid prolonged leaf wetness</div>
<div class="action"><span class="check">✓</span> Monitor plants regularly</div>
<div class="action"><span class="check">✓</span> Follow locally appropriate guidance</div>
</div>
</div>""",
        unsafe_allow_html=True,
    )


# Recent history
with b3:
    st.markdown(
"""<div class="card">
<div class="card-title">◷ Recent History</div>""",
        unsafe_allow_html=True,
    )

    if st.session_state.history:
        for item in st.session_state.history[:4]:
            status = "Healthy" if item["class"] == "healthy" else "Detected"

            st.markdown(
f"""<div style="padding:8px 0;border-bottom:1px solid #edf1ee;">
<b style="font-size:10px;">{item["name"]}</b>
<br>
<span style="font-size:9px;color:#708078;">
{item["confidence"]:.2f}% • {item["filename"]}
</span>
<span style="float:right;font-size:9px;color:#168447;font-weight:800;">
{status}
</span>
</div>""",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
"""<div class="info-box">
No scans yet.<br>
Upload your first leaf image.
</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
"""<div class="footer">
🌱 Protect your crops, improve your decisions, and build a smarter future with AI.
<br>
<b>Smart Farming, Better Future. 🌿</b>
<br>
Tomato Leaf Disease Detector • AI Powered Smart Farming
</div>""",
    unsafe_allow_html=True,
)
