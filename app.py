from __future__ import annotations

import json
import os
import threading
import traceback
from pathlib import Path
from typing import Any

import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


# =============================================================================
# Configuration
# =============================================================================

MODEL_PATH = os.getenv("MODEL_PATH", "tomato_leaf_model.pth")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Standard PlantVillage Tomato classes.
# If your trained model uses different class names, replace this list or set
# CLASS_NAMES_JSON to a JSON array in the environment.
DEFAULT_CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two_spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Healthy",
]

try:
    CLASS_NAMES = json.loads(
        os.getenv("CLASS_NAMES_JSON", json.dumps(DEFAULT_CLASS_NAMES))
    )
    if not isinstance(CLASS_NAMES, list) or not CLASS_NAMES:
        CLASS_NAMES = DEFAULT_CLASS_NAMES
except Exception:
    CLASS_NAMES = DEFAULT_CLASS_NAMES

_MODEL: Any | None = None
_MODEL_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


# =============================================================================
# Model loading
# =============================================================================

def build_model(num_classes: int) -> torch.nn.Module:
    """
    Build the same general type of image-classification model expected by the
    application.

    Default: ResNet-50.
    If your project was trained with another architecture, change this function
    to match that architecture before loading the .pth/.pt weights.
    """
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model() -> torch.nn.Module:
    if not Path(MODEL_PATH).exists():
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}. "
            "Upload/copy your trained tomato leaf model and set MODEL_PATH."
        )

    print(f"Loading tomato leaf model: {MODEL_PATH}")
    model = build_model(len(CLASS_NAMES))

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    # Support common checkpoint formats.
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Remove DataParallel prefix if present.
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
        cleaned_state_dict[new_key] = value

    try:
        model.load_state_dict(cleaned_state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "The model architecture/classes do not match this app. "
            "Update build_model() and/or CLASS_NAMES to match your training code.\n"
            f"Original error: {error}"
        ) from error

    model.to(DEVICE)
    model.eval()

    print(f"Device: {DEVICE}")
    print(f"Classes: {len(CLASS_NAMES)}")
    print("Tomato leaf disease model is ready.")
    return model


def get_model() -> torch.nn.Module:
    global _MODEL

    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                _MODEL = load_model()

    return _MODEL


# =============================================================================
# Image helpers
# =============================================================================

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image is None:
        raise ValueError("Please upload a tomato leaf image first.")
    return image.convert("RGB")


def extract_pixel_features(image: Image.Image) -> dict[str, float]:
    """Simple image-quality/visual statistics used as supplementary evidence."""
    rgb = np.asarray(ensure_rgb(image)).astype(np.float32)

    gray = (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    )

    brightness = float(gray.mean())
    contrast = float(gray.std())

    # Green dominance can be useful as a very simple supplementary observation.
    green_dominance = float(
        np.mean(rgb[:, :, 1] - (rgb[:, :, 0] + rgb[:, :, 2]) / 2.0)
    )

    return {
        "brightness": brightness,
        "contrast": contrast,
        "mean_red": float(rgb[:, :, 0].mean()),
        "mean_green": float(rgb[:, :, 1].mean()),
        "mean_blue": float(rgb[:, :, 2].mean()),
        "green_dominance": green_dominance,
    }


def pixel_features_to_text(features: dict[str, float]) -> str:
    brightness_level = (
        "low"
        if features["brightness"] < 70
        else "medium"
        if features["brightness"] < 180
        else "high"
    )

    contrast_level = (
        "low"
        if features["contrast"] < 30
        else "medium"
        if features["contrast"] < 70
        else "high"
    )

    green_level = (
        "low"
        if features["green_dominance"] < 5
        else "medium"
        if features["green_dominance"] < 25
        else "high"
    )

    return (
        f"Brightness: {brightness_level} ({features['brightness']:.2f})\n"
        f"Contrast: {contrast_level} ({features['contrast']:.2f})\n"
        f"Green dominance: {green_level} ({features['green_dominance']:.2f})\n"
        f"Average RGB: R={features['mean_red']:.1f}, "
        f"G={features['mean_green']:.1f}, B={features['mean_blue']:.1f}."
    )


# =============================================================================
# Prediction
# =============================================================================

def pretty_label(label: str) -> str:
    label = str(label)
    if "___" in label:
        label = label.split("___", 1)[1]

    label = label.replace("_", " ")
    return label


def predict_disease(
    image: Image.Image,
) -> tuple[str, dict[str, float], dict[str, Any]]:
    model = get_model()
    rgb_image = ensure_rgb(image)

    tensor = TRANSFORM(rgb_image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = F.softmax(logits, dim=1)[0]

    top_k = min(5, len(CLASS_NAMES))
    values, indices = torch.topk(probabilities, k=top_k)

    top_predictions = {}
    for value, index in zip(values.detach().cpu().tolist(),
                            indices.detach().cpu().tolist()):
        class_name = pretty_label(CLASS_NAMES[index])
        top_predictions[class_name] = float(value * 100.0)

    best_index = int(torch.argmax(probabilities).item())
    predicted_class = pretty_label(CLASS_NAMES[best_index])
    confidence = float(probabilities[best_index].item() * 100.0)

    features = extract_pixel_features(rgb_image)
    pixel_text = pixel_features_to_text(features)

    technical = {
        "model_path": MODEL_PATH,
        "device": str(DEVICE),
        "image_size": IMAGE_SIZE,
        "number_of_classes": len(CLASS_NAMES),
        "predicted_class": predicted_class,
        "confidence_percent": round(confidence, 2),
        "top_predictions": top_predictions,
        "pixel_features": features,
    }

    return predicted_class, top_predictions, technical



# =============================================================================
# Streamlit UI
# =============================================================================

st.set_page_config(
    page_title="Tomato Leaf Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(34,197,94,0.13), transparent 28%),
            radial-gradient(circle at 90% 8%, rgba(132,204,22,0.08), transparent 25%),
            linear-gradient(180deg, #07110b 0%, #09150d 55%, #07110b 100%);
        color: #f0fdf4;
    }

    .main-title {
        font-size: 54px;
        line-height: 1.05;
        font-weight: 900;
        letter-spacing: -2px;
        margin-bottom: 12px;
    }

    .main-title span {
        color: #86efac;
    }

    .subtitle {
        color: #b7c8bc;
        font-size: 18px;
        line-height: 1.7;
        max-width: 760px;
        margin-bottom: 28px;
    }

    .card {
        padding: 25px;
        border-radius: 20px;
        border: 1px solid rgba(74,222,128,0.22);
        background: rgba(15,30,20,0.78);
        margin-bottom: 20px;
    }

    .result-card {
        padding: 25px;
        border-radius: 18px;
        border: 1px solid rgba(74,222,128,0.30);
        background: linear-gradient(
            145deg,
            rgba(34,197,94,0.10),
            rgba(15,35,21,0.72)
        );
    }

    .result-label {
        color: #86efac;
        font-size: 14px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .result-name {
        color: #f0fdf4;
        font-size: 32px;
        font-weight: 900;
        margin: 5px 0 12px 0;
    }

    .confidence {
        color: #dcfce7;
        font-size: 17px;
        font-weight: 700;
    }

    .section-title {
        color: #f0fdf4;
        font-size: 28px;
        font-weight: 900;
        margin: 20px 0 12px 0;
    }

    .note {
        color: #9fb1a5;
        font-size: 13px;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="card">
        <div style="color:#86efac;font-weight:800;font-size:13px;">
            🌱 COMPUTER VISION FOR AGRICULTURE
        </div>
        <div class="main-title">
            Tomato Leaf<br><span>Disease Detector</span>
        </div>
        <div class="subtitle">
            Upload a tomato leaf image and let the trained AI model identify
            the most likely disease class with a confidence score and
            supporting image observations.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("### 🌿 Upload Tomato Leaf")
    uploaded_file = st.file_uploader(
        "Upload one clear tomato leaf image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded tomato leaf", use_container_width=True)

    detect = st.button(
        "🔍 Detect Disease",
        type="primary",
        use_container_width=True,
    )

with right:
    st.markdown("### 📊 Analysis Result")

    if uploaded_file is None:
        st.markdown(
            """
            <div class="result-card">
                <div class="result-label">Status</div>
                <div class="result-name">Ready for Analysis</div>
                <div class="note">
                    Upload a tomato leaf image and click Detect Disease.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif detect:
        try:
            with st.spinner("Analyzing tomato leaf..."):
                with _INFERENCE_LOCK:
                    prediction, top_predictions, technical = predict_disease(image)

            confidence = technical["confidence_percent"]

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted Disease</div>
                    <div class="result-name">{prediction}</div>
                    <div class="confidence">
                        Confidence: {confidence:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("### 🔝 Top Predictions")
            for label, score in top_predictions.items():
                st.progress(
                    min(max(score / 100.0, 0.0), 1.0),
                    text=f"{label}: {score:.2f}%",
                )

            st.markdown("### 🖼️ Image Observations")
            features = technical["pixel_features"]
            st.write(
                f"**Brightness:** {features['brightness']:.2f}  \n"
                f"**Contrast:** {features['contrast']:.2f}  \n"
                f"**Green dominance:** {features['green_dominance']:.2f}  \n"
                f"**Average RGB:** "
                f"R={features['mean_red']:.1f}, "
                f"G={features['mean_green']:.1f}, "
                f"B={features['mean_blue']:.1f}"
            )

            with st.expander("Advanced Technical Details"):
                st.json(technical)

        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.exception(error)

st.markdown(
    """
    <div class="card">
        <div class="section-title">Important Information</div>
        <div class="note">
            This application is a research/academic prototype. The AI
            prediction should not be treated as a definitive agricultural
            diagnosis. Image quality, dataset bias and model performance
            can affect the result. Confirm important treatment decisions
            with a qualified agricultural professional.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if __name__ == "__main__":
    # Streamlit runs this file directly; no explicit launch() is required.
    pass
