import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# ============================================================
# 🍅 TOMATO LEAF DISEASE DETECTOR
# Streamlit Professional UI
# ============================================================

st.set_page_config(
    page_title="Tomato Leaf Disease Detector",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.keras")

IMG_SIZE = (224, 224)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

# Exact class order from project notebook
CLASS_NAMES = [
    "Bacterial_spot",
    "Early_blight",
    "Late_blight",
    "Leaf_Miner",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites",
    "Verticillium_wilt",
    "healthy"
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
    "healthy": "Healthy Leaf"
}

DISEASE_INFO = {
    "Bacterial_spot": {
        "description": "A bacterial disease that can produce dark spots on tomato leaves.",
        "recommendations": [
            "Remove severely affected leaves.",
            "Improve plant spacing and airflow.",
            "Avoid unnecessary overhead watering."
        ]
    },

    "Early_blight": {
        "description": "A common fungal disease associated with dark lesions and concentric patterns on tomato leaves.",
        "recommendations": [
            "Remove severely infected leaves.",
            "Improve air circulation around plants.",
            "Avoid prolonged leaf wetness.",
            "Follow locally recommended disease-management practices."
        ]
    },

    "Late_blight": {
        "description": "A serious tomato disease that can spread rapidly under favorable environmental conditions.",
        "recommendations": [
            "Remove affected plant material.",
            "Improve ventilation around plants.",
            "Monitor nearby plants regularly.",
            "Use locally recommended management practices."
        ]
    },

    "Leaf_Miner": {
        "description": "Leaf miner damage commonly appears as winding trails or mines inside leaf tissue.",
        "recommendations": [
            "Remove heavily affected leaves.",
            "Inspect leaves regularly.",
            "Monitor plants for further activity."
        ]
    },

    "Leaf_Mold": {
        "description": "Leaf mold is a fungal disease that is commonly associated with humid conditions.",
        "recommendations": [
            "Improve ventilation.",
            "Reduce prolonged leaf wetness.",
            "Remove severely affected foliage."
        ]
    },

    "Septoria_leaf_spot": {
        "description": "A fungal leaf-spot disease that commonly produces numerous small lesions.",
        "recommendations": [
            "Remove affected foliage.",
            "Improve plant spacing.",
            "Keep leaves as dry as practical.",
            "Monitor plants regularly."
        ]
    },

    "Spider_mites": {
        "description": "Spider mites are tiny pests that can cause stippling and discoloration of tomato leaves.",
        "recommendations": [
            "Inspect the underside of leaves.",
            "Monitor plants frequently.",
            "Use an appropriate locally recommended management method."
        ]
    },

    "Verticillium_wilt": {
        "description": "A soil-borne disease that can cause yellowing, wilting and reduced plant vigor.",
        "recommendations": [
            "Remove affected plants where appropriate.",
            "Use crop rotation.",
            "Consider resistant varieties."
        ]
    },

    "healthy": {
        "description": "The model classified this image as a healthy tomato leaf.",
        "recommendations": [
            "Continue regular crop monitoring.",
            "Maintain good airflow.",
            "Keep plants healthy with proper crop management."
        ]
    }
}

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_ai_model():

    if not os.path.exists(MODEL_PATH):
        st.error(
            "❌ model.keras was not found.\n\n"
            "Make sure model.keras is in the same GitHub repository folder as app.py."
        )
        st.stop()

    model = load_model(
        MODEL_PATH,
        compile=False
    )

    return model


model = load_ai_model()

# ============================================================
# CLAHE PREPROCESSING
# Same preprocessing used in project validation pipeline
# ============================================================

def clahe_preprocessing(img):

    img = img.astype(np.uint8)

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge(
        (l, a, b)
    )

    img = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB
    )

    return img.astype(np.float32)


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    resized = image.resize(
        IMG_SIZE
    )

    image_array = np.asarray(
        resized
    ).astype(np.float32)

    image_array = clahe_preprocessing(
        image_array
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image):

    processed = preprocess_image(
        image
    )

    predictions = model.predict(
        processed,
        verbose=0
    )[0]

    top_indices = np.argsort(
        predictions
    )[::-1][:3]

    top3 = []

    for index in top_indices:

        class_name = CLASS_NAMES[index]

        confidence = (
            float(predictions[index]) * 100
        )

        top3.append({
            "class": class_name,
            "name": DISPLAY_NAMES[class_name],
            "confidence": confidence
        })

    return top3


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
"""
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

html,
body,
[class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f5f8f6;
}

/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {

    background:
    linear-gradient(
        180deg,
        #052d1f 0%,
        #063e28 55%,
        #052d1f 100%
    );

}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.sidebar-brand {
    padding: 15px 5px 25px;
}

.brand-logo {

    width: 55px;
    height: 55px;

    display: inline-flex;

    align-items: center;
    justify-content: center;

    border-radius: 16px;

    border: 2px solid #a5dd4d;

    font-size: 28px;

    margin-right: 10px;

}

.brand-name {

    display: inline-block;

    font-size: 19px;

    font-weight: 800;

    vertical-align: middle;

}

.brand-sub {

    margin-left: 67px;

    margin-top: -13px;

    color: #b6dd39 !important;

    font-size: 12px;

    font-weight: 700;

}

.side-info {

    margin-top: 30px;

    padding: 18px;

    border-radius: 17px;

    border:
    1px solid rgba(150,230,170,.2);

    background:
    rgba(255,255,255,.06);

    line-height: 1.7;

}

/* ================= HEADER ================= */

.main-title {

    font-size: 34px;

    font-weight: 800;

    color: #14261e;

    letter-spacing: -1.5px;

}

.main-subtitle {

    color: #708078;

    font-size: 13px;

    margin-bottom: 25px;

}

/* ================= CARD ================= */

.card {

    background: white;

    border: 1px solid #e3ebe6;

    border-radius: 20px;

    padding: 22px;

    box-shadow:
    0 14px 40px
    rgba(15,53,35,.07);

    margin-bottom: 18px;

}

.card-title {

    font-size: 16px;

    font-weight: 800;

    color: #15251e;

}

.card-subtitle {

    color: #708078;

    font-size: 11px;

    margin-top: 6px;

    margin-bottom: 18px;

}

/* ================= UPLOAD ================= */

[data-testid="stFileUploaderDropzone"] {

    border:
    2px dashed #63b878 !important;

    border-radius: 17px !important;

    background:
    linear-gradient(
        180deg,
        #fbfefc,
        #f3faf5
    ) !important;

    min-height: 230px;

}

[data-testid="stFileUploaderDropzone"] button {

    background:
    #168447 !important;

    color: white !important;

    border: none !important;

    border-radius: 9px !important;

}

/* ================= PREDICTION ================= */

.prediction-card {

    padding: 18px;

    border-radius: 16px;

    background:
    #f0f8f1;

    border:
    1px solid #dceede;

}

.prediction-danger {

    background:
    #fff1f2;

    border:
    1px solid #f2d5d7;

}

.prediction-title {

    font-size: 29px;

    font-weight: 800;

    color: #0b5a36;

    margin: 6px 0;

}

.prediction-danger
.prediction-title {

    color: #bd3f47;

}

.confidence {

    color: #168447;

    font-size: 24px;

    font-weight: 800;

}

/* ================= INSIGHTS ================= */

.insight {

    background: white;

    border: 1px solid #e3ebe6;

    border-radius: 17px;

    padding: 20px;

    min-height: 145px;

    box-shadow:
    0 10px 28px
    rgba(15,53,35,.05);

}

.insight-icon {

    font-size: 28px;

    margin-bottom: 10px;

}

.insight-title {

    font-size: 13px;

    font-weight: 800;

}

.insight-text {

    color: #708078;

    font-size: 10px;

    line-height: 1.6;

    margin-top: 6px;

}

/* ================= BOTTOM ================= */

.info-box {

    background: #f7faf8;

    border-radius: 12px;

    padding: 12px;

    margin-top: 9px;

    color: #52645a;

    font-size: 10px;

    line-height: 1.6;

}

.check {

    color: #1b9c4e;

    font-weight: 800;

}

/* ================= STATISTICS ================= */

.stat {

    background: white;

    border-radius: 16px;

    padding: 18px;

    border: 1px solid #e3ebe6;

}

.stat-number {

    font-size: 27px;

    font-weight: 800;

    color: #168447;

}

.stat-label {

    font-size: 10px;

    color: #708078;

}

/* ================= FOOTER ================= */

.footer {

    padding: 17px;

    border-radius: 15px;

    background:
    linear-gradient(
        90deg,
        #edf8ef,
        #f9fcf8
    );

    border:
    1px solid #dcebdc;

    color: #56675d;

    font-size: 11px;

    text-align: center;

    margin-top: 20px;

}

/* ================= BUTTON ================= */

.stButton > button {

    border-radius: 10px;

    font-weight: 700;

    min-height: 42px;

}

/* ================= IMAGE ================= */

[data-testid="stImage"] img {

    border-radius: 15px;

}

/* ================= MOBILE ================= */

@media(max-width: 900px) {

    .main-title {
        font-size: 27px;
    }

}

</style>
""",
unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">

            <span class="brand-logo">
                🌿
            </span>

            <span class="brand-name">
                Tomato Leaf
            </span>

            <div class="brand-sub">
                Disease Detector
            </div>

        </div>
        """,
        unsafe_allow_html=True
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
            "About Project"
        ],
        label_visibility="collapsed"
    )

    st.markdown(
        """
        <div class="side-info">

            <b>⚡ AI Powered Detection</b>

            <br><br>

            <span style="font-size:11px;opacity:.75;">
            Advanced deep learning model for
            accurate and reliable tomato leaf
            disease detection.
            </span>

            <br><br>

            🍅 🌿 🍅 🌿 🍅

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
        margin-top:40px;
        font-size:10px;
        opacity:.55;
        ">
        © 2026 Tomato Leaf Disease Detector
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "prediction" not in st.session_state:
    st.session_state.prediction = None


# ============================================================
# DISEASE GUIDE
# ============================================================

if menu == "Disease Guide":

    st.markdown(
        """
        <div class="main-title">
            Tomato Disease Guide
        </div>

        <div class="main-subtitle">
            Diseases and conditions supported by the trained AI model.
        </div>
        """,
        unsafe_allow_html=True
    )

    cols = st.columns(3)

    for i, class_name in enumerate(CLASS_NAMES):

        info = DISEASE_INFO[class_name]

        with cols[i % 3]:

            st.markdown(
                f"""
                <div class="card">

                    <div class="card-title">
                        {DISPLAY_NAMES[class_name]}
                    </div>

                    <div class="info-box">
                        {info["description"]}
                    </div>

                    <br>

                    <b style="font-size:11px;">
                    Recommended Actions
                    </b>

                    <div style="margin-top:8px;">

                    {"".join(
                        f'<div style="font-size:10px;margin:7px 0;">'
                        f'<span class="check">✓</span> {x}'
                        f'</div>'
                        for x in info["recommendations"]
                    )}

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.stop()


# ============================================================
# PREVENTION
# ============================================================

if menu == "Prevention Tips":

    st.markdown(
        """
        <div class="main-title">
            Prevention Tips
        </div>

        <div class="main-subtitle">
            General practices for maintaining healthier tomato plants.
        </div>
        """,
        unsafe_allow_html=True
    )

    tips = [
        ("🌬️", "Improve Air Circulation",
         "Keep sufficient spacing between plants to reduce prolonged humidity around leaves."),

        ("💧", "Manage Leaf Wetness",
         "Avoid unnecessary overhead watering and reduce prolonged leaf wetness."),

        ("🔍", "Monitor Regularly",
         "Inspect leaves frequently so unusual symptoms can be noticed early."),

        ("🌱", "Remove Affected Leaves",
         "Remove severely affected plant material where appropriate."),

        ("🔄", "Crop Rotation",
         "Use suitable crop rotation practices to help manage some soil-associated diseases."),

        ("🧤", "Maintain Hygiene",
         "Keep tools and growing areas clean and remove infected plant debris appropriately.")
    ]

    cols = st.columns(3)

    for i, (icon, title, description) in enumerate(tips):

        with cols[i % 3]:

            st.markdown(
                f"""
                <div class="card">

                    <div style="font-size:28px;">
                        {icon}
                    </div>

                    <div class="card-title">
                        {title}
                    </div>

                    <div class="info-box">
                        {description}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.stop()


# ============================================================
# ABOUT
# ============================================================

if menu == "About Project":

    st.markdown(
        """
        <div class="main-title">
            About the Project
        </div>

        <div class="main-subtitle">
            AI-powered tomato leaf disease classification system.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                🍅 Tomato Leaf Disease Detector
            </div>

            <div class="info-box">

            This application provides a web interface
            for the trained deep-learning model used in
            the project.

            <br><br>

            The project uses a DenseNet121-based
            architecture with global average pooling,
            global max pooling, concatenation,
            batch normalization, dense layers and
            a softmax classification output.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    stats = [
        ("224×224", "Input Size"),
        ("9", "Classes"),
        ("DenseNet121", "Architecture"),
        ("Softmax", "Output")
    ]

    for col, (number, label) in zip(
        [c1, c2, c3, c4],
        stats
    ):

        with col:

            st.markdown(
                f"""
                <div class="stat">

                    <div class="stat-number">
                        {number}
                    </div>

                    <div class="stat-label">
                        {label}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.stop()


# ============================================================
# STATISTICS
# ============================================================

if menu == "Statistics":

    st.markdown(
        """
        <div class="main-title">
            Statistics Overview
        </div>

        <div class="main-subtitle">
            Overview of predictions made during this session.
        </div>
        """,
        unsafe_allow_html=True
    )

    history = st.session_state.history

    total = len(history)

    diseases = sum(
        1 for x in history
        if x["class"] != "healthy"
    )

    healthy = sum(
        1 for x in history
        if x["class"] == "healthy"
    )

    avg_confidence = (
        np.mean(
            [x["confidence"] for x in history]
        )
        if history else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    values = [
        (total, "Total Predictions"),
        (diseases, "Diseases Detected"),
        (healthy, "Healthy Leaves"),
        (f"{avg_confidence:.2f}%", "Average Confidence")
    ]

    for col, (number, label) in zip(
        [c1, c2, c3, c4],
        values
    ):

        with col:

            st.markdown(
                f"""
                <div class="stat">

                    <div class="stat-number">
                        {number}
                    </div>

                    <div class="stat-label">
                        {label}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("### Prediction Distribution")

    if history:

        counts = {}

        for item in history:

            name = item["name"]

            counts[name] = (
                counts.get(name, 0) + 1
            )

        st.bar_chart(counts)

    else:

        st.info(
            "No predictions yet. Analyze a leaf first."
        )

    st.stop()


# ============================================================
# HISTORY
# ============================================================

if menu == "History":

    st.markdown(
        """
        <div class="main-title">
            Recent History
        </div>

        <div class="main-subtitle">
            Predictions made during this browser session.
        </div>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state.history:

        st.info(
            "No prediction history yet."
        )

    else:

        for item in st.session_state.history:

            status = (
                "Healthy"
                if item["class"] == "healthy"
                else "Detected"
            )

            st.markdown(
                f"""
                <div class="card">

                    <div class="card-title">
                        {item["name"]}
                    </div>

                    <div class="mini-muted">
                        {item["confidence"]:.2f}% confidence
                    </div>

                    <div class="info-box">
                        📄 {item["filename"]}
                        <br>
                        Status: <b>{status}</b>
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.stop()


# ============================================================
# MAIN DASHBOARD
# ============================================================

st.markdown(
    """
    <div class="main-title">
        AI-Powered Tomato Leaf Analysis 🌿
    </div>

    <div class="main-subtitle">
        Upload a leaf image and identify potential diseases with AI.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD + RESULT
# ============================================================

left, right = st.columns(
    [1, 1.15],
    gap="large"
)


# ---------------- LEFT ----------------

with left:

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                ⇧ Upload Tomato Leaf Image
            </div>

            <div class="card-subtitle">
                Upload a clear image of the tomato leaf
                to detect possible diseases.
            </div>

        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Drag & drop your leaf image here",
        type=["jpg", "jpeg", "png"],
        help="Maximum file size: 5 MB"
    )

    st.markdown(
        """
        <div style="
        color:#708078;
        font-size:10px;
        margin-top:8px;
        ">
        Supported formats: JPG, JPEG, PNG
        &nbsp; • &nbsp;
        Max size: 5MB
        </div>

        <div class="info-box">
        💡 <b>Tip:</b>
        Use a clear image in good lighting
        for best results.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------- RIGHT ----------------

with right:

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                📈 Prediction Result
            </div>

            <div class="card-subtitle">
                AI-assisted classification result
                from your trained model.
            </div>

        """,
        unsafe_allow_html=True
    )

    if uploaded_file is None:

        st.markdown(
            """
            <div style="
            min-height:300px;
            display:flex;
            align-items:center;
            justify-content:center;
            text-align:center;
            color:#87958d;
            font-size:12px;
            ">
                Upload an image to see
                the prediction result.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        if uploaded_file.size > MAX_FILE_SIZE:

            st.error(
                "Image size must be less than 5 MB."
            )

        else:

            try:

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                image_col, result_col = st.columns(
                    [1, 1]
                )

                with image_col:

                    st.image(
                        image,
                        use_container_width=True
                    )

                with result_col:

                    with st.spinner(
                        "Analyzing leaf..."
                    ):

                        top3 = predict_image(
                            image
                        )

                    best = top3[0]

                    is_healthy = (
                        best["class"] == "healthy"
                    )

                    css_class = (
                        ""
                        if is_healthy
                        else "prediction-danger"
                    )

                    status = (
                        "Healthy Leaf"
                        if is_healthy
                        else "Disease Detected"
                    )

                    st.markdown(
                        f"""
                        <div class="
                        prediction-card
                        {css_class}
                        ">

                            <div style="
                            font-size:10px;
                            font-weight:800;
                            color:#687970;
                            ">
                                ● {status}
                            </div>

                            <div class="
                            prediction-title
                            ">
                                {best["name"]}
                            </div>

                            <div class="mini-muted">
                                {best["class"]}
                            </div>

                            <br>

                            <div class="mini-muted">
                                Confidence Score
                            </div>

                            <div class="confidence">
                                {best["confidence"]:.2f}%
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.progress(
                        min(
                            best["confidence"] / 100,
                            1.0
                        )
                    )

                # ---------------- INFO ----------------

                info = DISEASE_INFO[
                    best["class"]
                ]

                st.markdown(
                    f"""
                    <div class="info-box">

                        <b>
                        About {best["name"]}
                        </b>

                        <br><br>

                        {info["description"]}

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ---------------- RECOMMENDATIONS ----------------

                st.markdown(
                    "#### Recommended Actions"
                )

                for recommendation in info[
                    "recommendations"
                ]:

                    st.markdown(
                        f"""
                        <div style="
                        font-size:10px;
                        margin:8px 0;
                        color:#52645a;
                        ">
                            <span class="check">
                            ✓
                            </span>
                            {recommendation}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # ---------------- TOP 3 ----------------

                st.markdown(
                    "#### Top-3 Predictions"
                )

                for rank, item in enumerate(
                    top3,
                    start=1
                ):

                    st.markdown(
                        f"""
                        <div style="
                        padding:9px 0;
                        border-bottom:
                        1px solid #edf1ee;
                        font-size:11px;
                        ">

                            <b>
                            {rank}.
                            {item["name"]}
                            </b>

                            <span style="
                            float:right;
                            color:#168447;
                            font-weight:800;
                            ">
                            {item["confidence"]:.2f}%
                            </span>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # ---------------- SAVE HISTORY ----------------

                history_entry = {
                    "name": best["name"],
                    "class": best["class"],
                    "confidence": best["confidence"],
                    "filename": uploaded_file.name
                }

                # Prevent repeated history entries
                already_exists = any(
                    x["filename"] == uploaded_file.name
                    and abs(
                        x["confidence"]
                        - best["confidence"]
                    ) < 0.001
                    for x in st.session_state.history
                )

                if not already_exists:

                    st.session_state.history.insert(
                        0,
                        history_entry
                    )

                    st.session_state.history = (
                        st.session_state.history[:10]
                    )

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# INSIGHTS
# ============================================================

st.markdown(
    "### Why AI-assisted detection?"
)

c1, c2, c3, c4 = st.columns(4)

insights = [

    (
        "🛡️",
        "Early Detection",
        "Detect possible leaf conditions early and support timely crop monitoring."
    ),

    (
        "🎯",
        "Better Yield",
        "Healthy plants can support better crop performance and productivity."
    ),

    (
        "🌱",
        "Save Resources",
        "Use AI-assisted information to support more informed crop decisions."
    ),

    (
        "📊",
        "Smart Farming",
        "Bring modern image classification technology into agricultural workflows."
    )
]

for col, (icon, title, text) in zip(
    [c1, c2, c3, c4],
    insights
):

    with col:

        st.markdown(
            f"""
            <div class="insight">

                <div class="insight-icon">
                    {icon}
                </div>

                <div class="insight-title">
                    {title}
                </div>

                <div class="insight-text">
                    {text}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# THREE BOTTOM CARDS
# ============================================================

st.markdown("### Analysis Overview")

b1, b2, b3 = st.columns(
    [1.1, 1, 1]
)

# ABOUT DISEASE

with b1:

    if st.session_state.prediction:

        current_class = (
            st.session_state.prediction
        )

    elif uploaded_file is not None:

        current_class = top3[0]["class"]

    else:

        current_class = "Early_blight"

    info = DISEASE_INFO[
        current_class
    ]

    st.markdown(
        f"""
        <div class="card">

            <div class="card-title">
                ⓘ About {DISPLAY_NAMES[current_class]}
            </div>

            <div class="info-box">
                {info["description"]}
            </div>

            <br>

            <div style="
            font-size:10px;
            line-height:1.8;
            color:#53655b;
            ">

            <span class="check">✓</span>
            AI-assisted image classification

            <br>

            <span class="check">✓</span>
            224 × 224 model input

            <br>

            <span class="check">✓</span>
            Confidence-based prediction

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# RECOMMENDED ACTIONS

with b2:

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                🛡️ Recommended Actions
            </div>

            <div style="
            margin-top:15px;
            font-size:10px;
            line-height:1.9;
            color:#52645a;
            ">

            <div>
            <span class="check">✓</span>
            Remove severely affected leaves
            </div>

            <div>
            <span class="check">✓</span>
            Ensure proper spacing and airflow
            </div>

            <div>
            <span class="check">✓</span>
            Avoid prolonged leaf wetness
            </div>

            <div>
            <span class="check">✓</span>
            Monitor plants regularly
            </div>

            <div>
            <span class="check">✓</span>
            Follow locally appropriate guidance
            </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# HISTORY

with b3:

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                ◷ Recent History
            </div>

        """,
        unsafe_allow_html=True
    )

    if st.session_state.history:

        for item in (
            st.session_state.history[:4]
        ):

            st.markdown(
                f"""
                <div style="
                padding:8px 0;
                border-bottom:
                1px solid #edf1ee;
                ">

                    <b style="
                    font-size:10px;
                    ">
                    {item["name"]}
                    </b>

                    <br>

                    <span style="
                    font-size:9px;
                    color:#708078;
                    ">
                    {item["confidence"]:.2f}%
                    </span>

                    <span style="
                    float:right;
                    font-size:9px;
                    color:#168447;
                    font-weight:700;
                    ">
                    Detected
                    </span>

                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.markdown(
            """
            <div class="info-box">
                No scans yet.<br>
                Upload your first leaf image.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# STATISTICS OVERVIEW
# ============================================================

st.markdown("### Statistics Overview")

history = st.session_state.history

total_predictions = len(history)

disease_count = sum(
    x["class"] != "healthy"
    for x in history
)

healthy_count = sum(
    x["class"] == "healthy"
    for x in history
)

average_confidence = (
    np.mean(
        [x["confidence"] for x in history]
    )
    if history
    else 0
)

s1, s2, s3, s4 = st.columns(4)

statistics = [

    (
        total_predictions,
        "Total Predictions"
    ),

    (
        disease_count,
        "Diseases Detected"
    ),

    (
        healthy_count,
        "Healthy Leaves"
    ),

    (
        f"{average_confidence:.2f}%",
        "Avg. Confidence"
    )
]

for col, (number, label) in zip(
    [s1, s2, s3, s4],
    statistics
):

    with col:

        st.markdown(
            f"""
            <div class="stat">

                <div class="stat-number">
                    {number}
                </div>

                <div class="stat-label">
                    {label}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        🌱 Protect your crops,
        improve your decisions,
        and build a smarter future with AI.

        <br><br>

        <b>
        Smart Farming, Better Future. 🌿
        </b>

        <br><br>

        Tomato Leaf Disease Detector •
        AI Powered Smart Farming

    </div>
    """,
    unsafe_allow_html=True
)
