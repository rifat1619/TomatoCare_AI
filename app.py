from __future__ import annotations

import json
import os
import threading
import traceback
from pathlib import Path
from typing import Any

import gradio as gr
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
        raise gr.Error("Please upload a tomato leaf image first.")
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
# Gradio callback
# =============================================================================

def run_prediction(
    image: Image.Image | None,
) -> tuple[str, str, str, dict[str, Any]]:
    if image is None:
        raise gr.Error("Please upload a tomato leaf image first.")

    try:
        with _INFERENCE_LOCK:
            prediction, top_predictions, technical = predict_disease(image)

        confidence = technical["confidence_percent"]

        result_markdown = (
            f"## Diagnosis: **{prediction}**\n\n"
            f"**Confidence:** {confidence:.2f}%\n\n"
            "> This prediction is generated by an AI research prototype. "
            "For agricultural treatment decisions, confirm the diagnosis with "
            "a qualified agricultural professional."
        )

        top_lines = "\n".join(
            f"{index}. **{label}** — {score:.2f}%"
            for index, (label, score) in enumerate(
                top_predictions.items(), start=1
            )
        )

        observations = (
            f"### Image Observations\n\n"
            f"{technical['pixel_features']}\n\n"
            f"### Top Predictions\n\n"
            f"{top_lines}"
        )

        return result_markdown, observations, prediction, technical

    except gr.Error:
        raise
    except Exception as error:
        traceback.print_exc()
        raise gr.Error(f"Prediction failed: {error}") from error


def clear_outputs() -> tuple[None, str, str, str, dict[str, Any]]:
    return (
        None,
        (
            "## Ready for Tomato Leaf Analysis\n\n"
            "Upload a tomato leaf image and click **Detect Disease**."
        ),
        "",
        "",
        {},
    )


# =============================================================================
# Custom CSS
# =============================================================================

CUSTOM_CSS = r"""
:root {
    --tld-bg: #07110b;
    --tld-bg2: #0b1710;
    --tld-card: rgba(18, 31, 22, 0.92);
    --tld-card2: rgba(24, 39, 28, 0.96);
    --tld-border: rgba(74, 222, 128, 0.25);
    --tld-text: #f0fdf4;
    --tld-muted: #b7c8bc;
    --tld-soft: #8fa296;
    --tld-accent: #22c55e;
    --tld-accent2: #86efac;
    --tld-yellow: #facc15;
}

html,
body,
.gradio-container {
    background:
        radial-gradient(circle at 10% 10%, rgba(34,197,94,0.15), transparent 27%),
        radial-gradient(circle at 88% 8%, rgba(132,204,22,0.10), transparent 24%),
        linear-gradient(180deg, #07110b 0%, #09150d 52%, #07110b 100%) !important;
    color: var(--tld-text) !important;
    width: 100%;
    max-width: 100%;
    overflow-x: hidden;
}

body {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system,
        BlinkMacSystemFont, "Segoe UI", sans-serif;
}

* {
    box-sizing: border-box;
}

.gradio-container {
    max-width: 1450px !important;
    margin: 0 auto !important;
    padding: 0 38px 30px 38px !important;
}

.tld-shell {
    width: min(1260px, 100%);
    margin: 0 auto;
}

.tld-navbar {
    min-height: 76px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    padding: 12px 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}

.tld-brand {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    color: var(--tld-text) !important;
    font-size: 22px;
    font-weight: 900;
}

.tld-logo {
    width: 40px;
    height: 40px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 13px;
    color: var(--tld-accent2);
    border: 1px solid rgba(74,222,128,0.5);
    background: linear-gradient(145deg, rgba(34,197,94,0.18), rgba(15,23,42,0.7));
    box-shadow: 0 0 25px rgba(34,197,94,0.18);
    font-size: 22px;
}

.tld-page {
    padding: 58px 0 25px 0;
}

.tld-hero {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(400px, 0.85fr);
    gap: 55px;
    align-items: center;
}

.tld-kicker {
    display: inline-flex;
    padding: 9px 15px;
    border-radius: 999px;
    color: #dcfce7;
    background: rgba(34,197,94,0.10);
    border: 1px solid rgba(74,222,128,0.22);
    font-size: 13px;
    font-weight: 850;
}

.tld-title {
    margin: 18px 0 18px 0;
    color: var(--tld-text) !important;
    font-size: clamp(42px, 5vw, 66px);
    line-height: 1.06;
    letter-spacing: -2px;
    font-weight: 950;
}

.tld-title span {
    color: var(--tld-accent2);
    text-shadow: 0 0 28px rgba(34,197,94,0.22);
}

.tld-subtitle {
    max-width: 700px;
    color: var(--tld-muted) !important;
    font-size: 18px;
    line-height: 1.75;
}

.tld-feature-card {
    padding: 34px;
    border-radius: 24px;
    border: 1px solid var(--tld-border);
    background:
        radial-gradient(circle at 90% 10%, rgba(34,197,94,0.13), transparent 30%),
        linear-gradient(145deg, rgba(20,36,25,0.96), rgba(9,20,13,0.96));
    box-shadow: 0 25px 65px rgba(0,0,0,0.35);
}

.tld-leaf-art {
    min-height: 310px;
    border-radius: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    background:
        radial-gradient(circle at 50% 42%, rgba(74,222,128,0.25), transparent 28%),
        radial-gradient(circle at 42% 62%, rgba(132,204,22,0.18), transparent 28%),
        linear-gradient(145deg, #0b1f12, #102b18);
    border: 1px solid rgba(74,222,128,0.16);
    font-size: 120px;
    filter: drop-shadow(0 25px 40px rgba(0,0,0,0.35));
}

.tld-section {
    padding: 35px 0 20px 0;
}

.tld-section-title {
    margin: 0 0 8px 0;
    color: var(--tld-text) !important;
    font-size: 34px;
    font-weight: 950;
}

.tld-section-subtitle {
    margin: 0 0 25px 0;
    color: var(--tld-muted) !important;
    line-height: 1.65;
}

.tld-info-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 18px;
}

.tld-info-card {
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.09);
    background: linear-gradient(145deg, rgba(22,38,27,0.86), rgba(12,24,16,0.90));
}

.tld-info-card h3 {
    margin: 0 0 9px 0;
    color: white !important;
    font-size: 18px;
    font-weight: 900;
}

.tld-info-card p {
    margin: 0;
    color: var(--tld-muted) !important;
    line-height: 1.65;
}

.tld-detector {
    padding: 28px;
    border-radius: 24px;
    border: 1px solid rgba(74,222,128,0.22);
    background:
        radial-gradient(circle at 0% 0%, rgba(34,197,94,0.10), transparent 28%),
        linear-gradient(145deg, rgba(20,34,24,0.96), rgba(10,20,13,0.96));
    box-shadow: 0 25px 65px rgba(0,0,0,0.30);
}

.tld-card {
    height: 100%;
    padding: 23px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.09);
    background: rgba(10,20,13,0.72);
}

.tld-card-title {
    margin: 0 0 7px 0;
    padding-left: 11px;
    border-left: 4px solid var(--tld-accent);
    color: var(--tld-text) !important;
    font-size: 20px;
    font-weight: 900;
}

.tld-card-subtitle {
    margin: 0 0 17px 0;
    color: var(--tld-muted) !important;
    font-size: 13px;
    line-height: 1.6;
}

#leaf-upload {
    min-height: 300px !important;
    border: 1.8px dashed rgba(74,222,128,0.65) !important;
    border-radius: 17px !important;
    background: rgba(5,14,8,0.80) !important;
    overflow: hidden !important;
}

.tld-detect-button {
    min-height: 50px !important;
    border-radius: 13px !important;
    border: 0 !important;
    color: white !important;
    background: linear-gradient(135deg, #16a34a, #22c55e) !important;
    font-size: 15px !important;
    font-weight: 900 !important;
    box-shadow: 0 14px 30px rgba(34,197,94,0.22) !important;
}

.tld-clear-button {
    min-height: 50px !important;
    border-radius: 13px !important;
    background: rgba(5,14,8,0.50) !important;
    color: var(--tld-text) !important;
    border: 1px solid rgba(74,222,128,0.30) !important;
    font-weight: 850 !important;
}

#result-card {
    min-height: 150px;
    padding: 23px !important;
    border: 1px solid rgba(74,222,128,0.30) !important;
    border-radius: 17px !important;
    background:
        radial-gradient(circle at 100% 0%, rgba(34,197,94,0.12), transparent 35%),
        linear-gradient(145deg, rgba(34,197,94,0.09), rgba(15,35,21,0.68)) !important;
    color: var(--tld-text) !important;
}

#result-card h2 {
    margin-top: 0 !important;
    color: var(--tld-text) !important;
    font-size: 28px !important;
}

.tld-output textarea,
.tld-output input {
    background: rgba(5,14,8,0.82) !important;
    color: var(--tld-text) !important;
    border-color: rgba(255,255,255,0.10) !important;
    border-radius: 14px !important;
}

.tld-output label,
.tld-output label span {
    color: var(--tld-accent2) !important;
    font-weight: 850 !important;
}

.tld-tech {
    margin-top: 18px;
    border-radius: 16px !important;
    background: rgba(8,18,11,0.90) !important;
    color: var(--tld-text) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
}

@media (max-width: 900px) {
    .gradio-container {
        padding: 0 18px 24px 18px !important;
    }

    .tld-hero {
        grid-template-columns: 1fr;
    }

    .tld-info-grid {
        grid-template-columns: 1fr;
    }
}
"""


# =============================================================================
# Gradio UI
# =============================================================================

with gr.Blocks(
    title="Tomato Leaf Disease Detector",
    theme=gr.themes.Base(),
    css=CUSTOM_CSS,
) as demo:

    # -------------------------------------------------------------------------
    # Home
    # -------------------------------------------------------------------------
    with gr.Group():
        gr.HTML(
            """
            <main class="tld-shell tld-page">
                <div class="tld-navbar">
                    <div class="tld-brand">
                        <div class="tld-logo">🌿</div>
                        Tomato Leaf AI
                    </div>
                    <div style="color:#b7c8bc;font-size:13px;font-weight:800;">
                        AI Disease Detection
                    </div>
                </div>

                <section class="tld-hero">
                    <div>
                        <div class="tld-kicker">🌱 Computer Vision for Agriculture</div>
                        <h1 class="tld-title">
                            Tomato Leaf<br><span>Disease Detector</span>
                        </h1>
                        <p class="tld-subtitle">
                            Upload a tomato leaf image and let the trained AI model
                            identify the most likely disease class with a confidence
                            score and supporting image observations.
                        </p>
                    </div>

                    <div class="tld-feature-card">
                        <div class="tld-leaf-art">🍃</div>
                    </div>
                </section>

                <section class="tld-section">
                    <h2 class="tld-section-title">Smart Leaf Analysis</h2>
                    <p class="tld-section-subtitle">
                        The application combines a trained image-classification
                        model with a clean, research-oriented interface.
                    </p>

                    <div class="tld-info-grid">
                        <div class="tld-info-card">
                            <h3>Image Classification</h3>
                            <p>
                                Processes the uploaded tomato leaf image and predicts
                                the most likely disease category.
                            </p>
                        </div>

                        <div class="tld-info-card">
                            <h3>Confidence Score</h3>
                            <p>
                                Shows the model's confidence and the top alternative
                                predictions for easier interpretation.
                            </p>
                        </div>

                        <div class="tld-info-card">
                            <h3>Visual Evidence</h3>
                            <p>
                                Provides basic brightness, contrast and color
                                observations as supplementary image evidence.
                            </p>
                        </div>
                    </div>
                </section>
            </main>
            """
        )

    # -------------------------------------------------------------------------
    # Detector
    # -------------------------------------------------------------------------
    with gr.Group(elem_classes=["tld-shell", "tld-page"]):
        with gr.Group(elem_classes=["tld-detector"]):
            with gr.Row(equal_height=True):

                with gr.Column(scale=5):
                    with gr.Group(elem_classes=["tld-card"]):
                        gr.HTML(
                            """
                            <h3 class="tld-card-title">Upload Tomato Leaf</h3>
                            <p class="tld-card-subtitle">
                                Upload one clear tomato leaf image for disease
                                classification.
                            </p>
                            """
                        )

                        image_input = gr.Image(
                            type="pil",
                            label="Tomato leaf image",
                            show_label=False,
                            height=300,
                            sources=["upload", "clipboard", "webcam"],
                            elem_id="leaf-upload",
                        )

                        with gr.Row():
                            predict_button = gr.Button(
                                "Detect Disease",
                                variant="primary",
                                scale=2,
                                elem_classes=["tld-detect-button"],
                            )

                            clear_button = gr.Button(
                                "Clear",
                                variant="secondary",
                                scale=1,
                                elem_classes=["tld-clear-button"],
                            )

                with gr.Column(scale=6):
                    with gr.Group(elem_classes=["tld-card"]):
                        gr.HTML(
                            """
                            <h3 class="tld-card-title">Analysis Result</h3>
                            <p class="tld-card-subtitle">
                                The predicted disease, confidence and supporting
                                observations will appear below.
                            </p>
                            """
                        )

                        result_output = gr.Markdown(
                            value=(
                                "## Ready for Tomato Leaf Analysis\n\n"
                                "Upload a tomato leaf image and click "
                                "**Detect Disease**."
                            ),
                            elem_id="result-card",
                        )

                        prediction_output = gr.Textbox(
                            label="Predicted disease",
                            placeholder="Predicted disease will appear here.",
                            interactive=False,
                            elem_classes=["tld-output"],
                        )

                        observations_output = gr.Textbox(
                            label="Analysis observations",
                            placeholder=(
                                "Top predictions and image-level observations "
                                "will appear here."
                            ),
                            lines=12,
                            max_lines=18,
                            interactive=False,
                            elem_classes=["tld-output"],
                        )

            with gr.Accordion(
                "Advanced Technical Details",
                open=False,
                elem_classes=["tld-tech"],
            ):
                technical_output = gr.JSON(label="Technical output")

    # -------------------------------------------------------------------------
    # About
    # -------------------------------------------------------------------------
    gr.HTML(
        """
        <main class="tld-shell tld-section">
            <div class="tld-info-card">
                <h3>Important Information</h3>
                <p>
                    This application is a research/academic prototype. The AI
                    prediction should not be treated as a definitive agricultural
                    diagnosis. Image quality, dataset bias and model performance
                    can affect the result.
                </p>
            </div>
        </main>
        """
    )

    predict_button.click(
        fn=run_prediction,
        inputs=[image_input],
        outputs=[
            result_output,
            observations_output,
            prediction_output,
            technical_output,
        ],
    )

    clear_button.click(
        fn=clear_outputs,
        inputs=[],
        outputs=[
            image_input,
            result_output,
            prediction_output,
            observations_output,
            technical_output,
        ],
    )


if __name__ == "__main__":
    demo.launch()
