import os
from io import BytesIO

import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model


# =========================================================
# TOMATO LEAF DISEASE DETECTOR
# Streamlit Version
# =========================================================

st.set_page_config(
    page_title="Tomato Leaf Disease Detector",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.keras")

IMG_SIZE = (224, 224)
MAX_FILE_SIZE = 5 * 1024 * 1024


# Exact class order from the project notebook
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
    "Bacterial_spot": (
        "Bacterial Spot is a bacterial disease that can produce dark spots "
        "on tomato leaves.",
        "Inspect affected leaves and improve airflow around the plants."
    ),

    "Early_blight": (
        "Early Blight is a common fungal disease associated with dark lesions "
        "and concentric patterns.",
        "Remove severely affected leaves and improve air circulation."
    ),

    "Late_blight": (
        "Late Blight is a serious tomato disease that can spread rapidly "
        "under favorable conditions.",
        "Remove affected plant material and follow appropriate crop-management guidance."
    ),

    "Leaf_Miner": (
        "Leaf Miner damage appears as winding trails or mines inside leaf tissue.",
        "Monitor leaves regularly and remove heavily affected foliage."
    ),

    "Leaf_Mold": (
        "Leaf Mold is a fungal disease commonly associated with humid conditions.",
        "Improve ventilation and reduce prolonged leaf wetness."
    ),

    "Septoria_leaf_spot": (
        "Septoria Leaf Spot is a fungal disease that produces multiple "
        "small lesions on tomato leaves.",
        "Remove affected leaves and improve spacing and airflow."
    ),

    "Spider_mites": (
        "Spider Mites are tiny pests that can cause stippling and discoloration "
        "of tomato leaves.",
        "Inspect the underside of leaves and monitor the plant closely."
    ),

    "Verticillium_wilt": (
        "Verticillium Wilt is a soil-borne disease that may cause yellowing, "
        "wilting and reduced plant vigor.",
        "Use good sanitation and suitable crop-rotation practices."
    ),

    "healthy": (
        "The model classified the uploaded image as a healthy tomato leaf.",
        "Continue regular monitoring and good crop-management practices."
    )
}


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* ---------- General ---------- */

    .stApp {
        background-color: #f5f8f6;
    }

    html, body, [class*="css"] {
        font-family: Arial, sans-serif;
    }

    /* ---------- Sidebar ---------- */

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #052d1f 0%,
            #064129 55%,
            #052d1f 100%
        );
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .brand-box {
        text-align: center;
        padding: 10px 0 25px 0;
    }

    .brand-icon {
        font-size: 45px;
    }

    .brand-title {
        font-size: 22px;
        font-weight: 800;
        margin-top: 5px;
    }

    .brand-subtitle {
        color: #b6dd39 !important;
        font-size: 12px;
        font-weight: 700;
        margin-top: 4px;
    }

    .sidebar-info {
        margin-top: 30px;
        padding: 18px;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.13);
        border-radius: 15px;
        line-height: 1.6;
    }

    /* ---------- Main Header ---------- */

    .main-title {
        font-size: 36px;
        font-weight: 800;
        color: #15251e;
        margin-bottom: 5px;
    }

    .main-subtitle {
        color: #708078;
        font-size: 14px;
        margin-bottom: 30px;
    }

    /* ---------- Cards ---------- */

    .custom-card {
        background: white;
        border: 1px solid #e1ebe4;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    .card-title {
        font-size: 18px;
        font-weight: 800;
        color: #15251e;
        margin-bottom: 6px;
    }

    .card-subtitle {
        color: #748078;
        font-size: 12px;
        margin-bottom: 18px;
    }

    /* ---------- Upload ---------- */

    [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #68b97e !important;
        background: #f7fcf8 !important;
        border-radius: 16px !important;
        padding: 20px !important;
    }

    [data-testid="stFileUploaderDropzone"] button {
        background: #168447 !important;
        color: white !important;
        border: none !important;
        border-radius: 9px !important;
    }

    /* ---------- Prediction ---------- */

    .prediction-box {
        background: #f0f9f2;
        border: 1px solid #d7ebda;
        border-radius: 16px;
        padding: 20px;
    }

    .prediction-box-danger {
        background: #fff3f3;
        border: 1px solid #f1d2d4;
        border-radius: 16px;
        padding: 20px;
    }

    .prediction-status {
        color: #168447;
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
    }

    .prediction-status-danger {
        color: #d4464e;
        font-size: 11px;
        font-weight: 800;
        text-transform: uppercase;
    }

    .prediction-name {
        color: #0b5a36;
        font-size: 30px;
        font-weight: 800;
        margin: 7px 0;
    }

    .prediction-name-danger {
        color: #ba3c44;
        font-size: 30px;
        font-weight: 800;
        margin: 7px 0;
    }

    .confidence {
        color: #168447;
        font-size: 25px;
        font-weight: 800;
    }

    .small-text {
        color: #708078;
        font-size: 11px;
    }

    /* ---------- Info cards ---------- */

    .info-card {
        background: white;
        border: 1px solid #e1ebe4;
        border-radius: 16px;
        padding: 20px;
        height: 100%;
        box-shadow: 0 8px 24px rgba(0,0,0,0.04);
    }

    .info-icon {
        font-size: 25px;
        margin-bottom: 8px;
    }

    .info-title {
        font-size: 13px;
        font-weight: 800;
        color: #15251e;
    }

    .info-text {
        font-size: 10px;
        color: #708078;
        line-height: 1.6;
        margin-top: 6px;
    }

    /* ---------- Guide ---------- */

    .guide-card {
        background: #f8fbf9;
        border: 1px solid #e1ebe4;
        border-radius: 14px;
        padding: 15px;
        margin-bottom: 12px;
    }

    .guide-title {
        font-size: 13px;
        font-weight: 800;
        color: #0b5a36;
        margin-bottom: 5px;
    }

    .guide-text {
        color: #65766d;
        font-size: 10px;
        line-height: 1.6;
    }

    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #75847c;
        font-size: 10px;
        padding: 25px 0 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource(show_spinner=False)
def load_ai_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "model.keras not found. "
            "Please keep model.keras in the same folder as app.py."
        )

    model = load_model(
        MODEL_PATH,
        compile=False
    )

    # Safety check
    output_units = model.output_shape[-1]

    if output_units != len(CLASS_NAMES):
        raise ValueError(
            f"Model output has {output_units} classes, "
            f"but this project expects {len(CLASS_NAMES)} classes."
        )

    return model


# =========================================================
# CLAHE PREPROCESSING
# Same validation preprocessing from notebook
# =========================================================

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


# =========================================================
# PREPROCESS IMAGE
# =========================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        IMG_SIZE
    )

    image_array = np.asarray(
        image
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


# =========================================================
# PREDICTION
# =========================================================

def predict_image(image, model):

    processed_image = preprocess_image(
        image
    )

    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]

    top_indices = np.argsort(
        predictions
    )[::-1][:3]

    results = []

    for index in top_indices:

        results.append(
            {
                "class": CLASS_NAMES[index],
                "name": DISPLAY_NAMES[CLASS_NAMES[index]],
                "confidence": float(
                    predictions[index] * 100
                )
            }
        )

    return results


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand-box">
            <div class="brand-icon">🌿</div>
            <div class="brand-title">
                Tomato Leaf
            </div>
            <div class="brand-subtitle">
                DISEASE DETECTOR
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🏠 Navigation")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Disease Guide",
            "About Project"
        ],
        label_visibility="collapsed"
    )

    st.markdown(
        """
        <div class="sidebar-info">
            <b>🤖 AI Powered Detection</b>
            <br><br>
            Deep learning based tomato leaf
            disease classification using the
            trained DenseNet121 model.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# PAGE: DISEASE GUIDE
# =========================================================

if page == "Disease Guide":

    st.markdown(
        '<div class="main-title">Disease Guide 📖</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'Diseases supported by the trained tomato leaf classification model.'
        '</div>',
        unsafe_allow_html=True
    )

    columns = st.columns(3)

    for i, class_name in enumerate(CLASS_NAMES):

        disease_name = DISPLAY_NAMES[class_name]

        description, recommendation = DISEASE_INFO[class_name]

        with columns[i % 3]:

            st.markdown(
                f"""
                <div class="guide-card">

                    <div class="guide-title">
                        {disease_name}
                    </div>

                    <div class="guide-text">
                        {description}
                        <br><br>
                        <b>Recommended:</b>
                        {recommendation}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.stop()


# =========================================================
# PAGE: ABOUT PROJECT
# =========================================================

if page == "About Project":

    st.markdown(
        '<div class="main-title">About Project 🌱</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'AI-powered Tomato Leaf Disease Detection System'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="custom-card">

            <div class="card-title">
                🍅 Tomato Leaf Disease Detector
            </div>

            <div class="card-subtitle">
                Deep Learning Based Image Classification
            </div>

            <p style="color:#65766d;font-size:12px;line-height:1.8;">

                This application uses the trained model supplied with the
                project to classify tomato leaf images into nine different
                categories.

            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Input Size",
            "224 × 224"
        )

    with c2:
        st.metric(
            "Classes",
            "9"
        )

    with c3:
        st.metric(
            "Architecture",
            "DenseNet121"
        )

    st.markdown("### Supported Classes")

    st.write(
        ", ".join(
            DISPLAY_NAMES[x]
            for x in CLASS_NAMES
        )
    )

    st.stop()


# =========================================================
# DASHBOARD
# =========================================================

st.markdown(
    '<div class="main-title">Welcome back! 👋</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Detect tomato leaf diseases with the power of AI.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

try:

    model = load_ai_model()

except Exception as e:

    st.error(
        f"Model loading failed: {e}"
    )

    st.stop()


# =========================================================
# MAIN TWO-COLUMN AREA
# =========================================================

left_column, right_column = st.columns(
    [1, 1.08],
    gap="large"
)


# =========================================================
# UPLOAD SECTION
# =========================================================

with left_column:

    st.markdown(
        """
        <div class="custom-card">

            <div class="card-title">
                📤 Upload Tomato Leaf
            </div>

            <div class="card-subtitle">
                Upload a clear leaf image for AI-powered disease detection.
            </div>

        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Drag and drop your image here",
        type=["jpg", "jpeg", "png"],
        help="Maximum image size: 5 MB"
    )

    st.markdown(
        """
        <div class="small-text">
            Supported formats: JPG, JPEG, PNG
            &nbsp; • &nbsp;
            Maximum size: 5 MB
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="guide-card">
            💡 <b>Tip:</b>
            Use a clear image with good lighting and keep the leaf visible.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# =========================================================
# RESULT SECTION
# =========================================================

with right_column:

    st.markdown(
        """
        <div class="custom-card">

            <div class="card-title">
                📊 Prediction Result
            </div>

            <div class="card-subtitle">
                Result generated by the trained AI model.
            </div>

        """,
        unsafe_allow_html=True
    )

    if uploaded_file is None:

        st.info(
            "Upload a tomato leaf image to start analysis."
        )

    else:

        if uploaded_file.size > MAX_FILE_SIZE:

            st.error(
                "The image is larger than 5 MB."
            )

            st.stop()

        try:

            image = Image.open(
                BytesIO(
                    uploaded_file.getvalue()
                )
            ).convert("RGB")

            # Display image
            st.image(
                image,
                caption=uploaded_file.name,
                use_container_width=True
            )

            # Predict
            with st.spinner(
                "Analyzing tomato leaf..."
            ):

                results = predict_image(
                    image,
                    model
                )

            best = results[0]

            is_healthy = (
                best["class"] == "healthy"
            )

            if is_healthy:

                st.markdown(
                    f"""
                    <div class="prediction-box">

                        <div class="prediction-status">
                            ● Healthy Leaf
                        </div>

                        <div class="prediction-name">
                            {best["name"]}
                        </div>

                        <div class="small-text">
                            AI classification result
                        </div>

                        <br>

                        <div class="small-text">
                            Confidence Score
                        </div>

                        <div class="confidence">
                            {best["confidence"]:.2f}%
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="prediction-box-danger">

                        <div class="prediction-status-danger">
                            ● Condition Detected
                        </div>

                        <div class="prediction-name-danger">
                            {best["name"]}
                        </div>

                        <div class="small-text">
                            AI classification result
                        </div>

                        <br>

                        <div class="small-text">
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

            # Information
            description, recommendation = DISEASE_INFO[
                best["class"]
            ]

            st.markdown(
                f"""
                <div class="guide-card">

                    <b>About this result</b>
                    <br><br>

                    {description}

                    <br><br>

                    <b>Recommended next step</b>
                    <br><br>

                    {recommendation}

                </div>
                """,
                unsafe_allow_html=True
            )

            # TOP 3
            with st.expander(
                "🔎 View Top-3 Predictions"
            ):

                for rank, item in enumerate(
                    results,
                    start=1
                ):

                    st.write(
                        f"**{rank}. {item['name']}**"
                        f" — {item['confidence']:.2f}%"
                    )

        except Exception as e:

            st.error(
                f"Could not analyze the image: {e}"
            )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# =========================================================
# BENEFITS
# =========================================================

st.markdown(
    "### Why AI-assisted detection?"
)

benefit_columns = st.columns(4)

benefits = [
    (
        "🛡️",
        "Early Detection",
        "Identify possible leaf conditions early."
    ),
    (
        "🎯",
        "Better Decisions",
        "Use AI screening to support crop management."
    ),
    (
        "🌱",
        "Save Resources",
        "Support informed agricultural decisions."
    ),
    (
        "📊",
        "Smart Farming",
        "Bring AI image classification into farming."
    )
]


for col, benefit in zip(
    benefit_columns,
    benefits
):

    with col:

        icon, title, text = benefit

        st.markdown(
            f"""
            <div class="info-card">

                <div class="info-icon">
                    {icon}
                </div>

                <div class="info-title">
                    {title}
                </div>

                <div class="info-text">
                    {text}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# SUPPORTED CLASSES
# =========================================================

st.markdown(
    "### Supported Disease Classes"
)

class_columns = st.columns(3)

for i, class_name in enumerate(CLASS_NAMES):

    with class_columns[i % 3]:

        st.markdown(
            f"""
            <div class="guide-card">

                <div class="guide-title">
                    {DISPLAY_NAMES[class_name]}
                </div>

                <div class="guide-text">
                    Model-supported class
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

        🌿 Tomato Leaf Disease Detector
        <br>
        AI Powered • Smart Farming • Deep Learning

    </div>
    """,
    unsafe_allow_html=True
)
