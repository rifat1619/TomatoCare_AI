import os
import base64
from io import BytesIO

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, jsonify, render_template_string, request
from tensorflow.keras.models import load_model

# ============================================================
# Tomato Leaf Disease Detector
# Model: DenseNet121_Advanced
# Input: 224 x 224 RGB
# Preprocessing: CLAHE + /255
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.keras")

IMG_SIZE = (224, 224)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# IMPORTANT:
# This is the exact class order printed by the training notebook.
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

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

# Load once when the Flask server starts.
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"model.keras was not found at: {MODEL_PATH}\n"
        "Put model.keras in the same folder as app.py."
    )

print("Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("Model loaded successfully.")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def clahe_preprocessing(img):
    """
    Same CLAHE preprocessing used in the project notebook.
    Notebook:
        RGB -> LAB
        CLAHE on L channel
        LAB -> RGB
        float32
        rescale 1./255
    """
    img = img.astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return img.astype(np.float32)


def preprocess_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    original = image.copy()

    image = image.resize(IMG_SIZE)
    img_array = np.asarray(image).astype(np.float32)

    # Match notebook validation/test pipeline.
    img_array = clahe_preprocessing(img_array)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return original, img_array


def predict_image(image_bytes):
    original, input_tensor = preprocess_image(image_bytes)

    probabilities = model.predict(input_tensor, verbose=0)[0]

    # Safety check in case a wrong model file is supplied.
    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            f"Model returned {len(probabilities)} classes, "
            f"but this project expects {len(CLASS_NAMES)}."
        )

    top_indices = np.argsort(probabilities)[::-1][:3]

    top3 = [
        {
            "label": DISPLAY_NAMES[CLASS_NAMES[i]],
            "raw_label": CLASS_NAMES[i],
            "confidence": round(float(probabilities[i]) * 100, 2),
        }
        for i in top_indices
    ]

    best = top3[0]

    # Preview image as base64 for the UI.
    preview = BytesIO()
    original.thumbnail((1000, 1000))
    original.save(preview, format="JPEG", quality=90)
    image_b64 = base64.b64encode(preview.getvalue()).decode("utf-8")

    return {
        "label": best["label"],
        "raw_label": best["raw_label"],
        "confidence": best["confidence"],
        "top3": top3,
        "image": f"data:image/jpeg;base64,{image_b64}",
    }


HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tomato Leaf Disease Detector</title>

    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link
        href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
        rel="stylesheet"
    >

    <style>
        :root {
            --green-950: #052b1d;
            --green-900: #073d27;
            --green-800: #0b5a36;
            --green-700: #168447;
            --green-600: #20a052;
            --green-500: #2dbb62;
            --green-100: #e7f7ed;
            --green-50: #f4fbf6;

            --ink: #15251e;
            --muted: #708078;
            --border: #e3ebe6;
            --bg: #f5f8f6;
            --white: #ffffff;

            --danger: #df4c55;
            --danger-bg: #fff0f1;
            --warning: #d68b19;
            --shadow: 0 14px 40px rgba(15, 53, 35, .08);
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: var(--bg);
            color: var(--ink);
        }

        button, input {
            font: inherit;
        }

        .app {
            min-height: 100vh;
            display: flex;
        }

        /* ---------------- Sidebar ---------------- */
        .sidebar {
            width: 270px;
            flex: 0 0 270px;
            min-height: 100vh;
            padding: 28px 18px 18px;
            color: white;
            background:
                radial-gradient(circle at 20% 20%, rgba(45,187,98,.14), transparent 28%),
                linear-gradient(180deg, #052d1f 0%, #063e28 52%, #052d1f 100%);
            display: flex;
            flex-direction: column;
            position: sticky;
            top: 0;
            height: 100vh;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 0 8px 28px;
        }

        .brand-icon {
            width: 52px;
            height: 52px;
            border: 2px solid #a5dd4d;
            border-radius: 16px;
            display: grid;
            place-items: center;
            font-size: 25px;
            background: rgba(255,255,255,.04);
        }

        .brand-title {
            font-weight: 800;
            font-size: 18px;
            line-height: 1.15;
        }

        .brand-subtitle {
            color: #b6dd39;
            font-size: 12px;
            font-weight: 700;
            margin-top: 4px;
        }

        .nav {
            display: grid;
            gap: 7px;
        }

        .nav-item {
            border: 0;
            background: transparent;
            color: rgba(255,255,255,.82);
            border-radius: 12px;
            padding: 13px 14px;
            display: flex;
            align-items: center;
            gap: 13px;
            cursor: pointer;
            text-align: left;
            transition: .2s ease;
        }

        .nav-item:hover,
        .nav-item.active {
            background: linear-gradient(135deg, #159c4d, #11723b);
            color: white;
            box-shadow: 0 8px 22px rgba(0,0,0,.14);
        }

        .nav-icon {
            width: 24px;
            text-align: center;
            font-size: 18px;
        }

        .nav-text {
            font-size: 13px;
            font-weight: 700;
        }

        .sidebar-card {
            margin-top: auto;
            padding: 20px;
            border: 1px solid rgba(153, 229, 166, .18);
            border-radius: 18px;
            background: rgba(255,255,255,.06);
        }

        .sidebar-card h4 {
            margin: 0 0 10px;
            font-size: 14px;
        }

        .sidebar-card p {
            margin: 0;
            color: rgba(255,255,255,.70);
            font-size: 12px;
            line-height: 1.65;
        }

        .sidebar-footer {
            margin-top: 18px;
            padding: 0 4px;
            color: rgba(255,255,255,.48);
            font-size: 10px;
            line-height: 1.6;
        }

        /* ---------------- Main ---------------- */
        .main {
            width: 100%;
            min-width: 0;
            padding: 28px 30px 34px;
        }

        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 18px;
            margin-bottom: 24px;
        }

        .eyebrow {
            font-size: 12px;
            color: var(--green-700);
            font-weight: 800;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin-bottom: 7px;
        }

        .topbar h1 {
            margin: 0;
            font-size: clamp(25px, 3vw, 34px);
            letter-spacing: -.04em;
        }

        .topbar p {
            margin: 8px 0 0;
            color: var(--muted);
            font-size: 13px;
        }

        .user-box {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .theme {
            border: 1px solid var(--border);
            background: white;
            border-radius: 999px;
            width: 50px;
            height: 38px;
            cursor: pointer;
        }

        .avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            color: white;
            background: linear-gradient(145deg, #218d4d, #06452b);
            font-weight: 800;
        }

        .user-info strong {
            display: block;
            font-size: 13px;
        }

        .user-info span {
            color: var(--muted);
            font-size: 11px;
        }

        /* ---------------- Cards ---------------- */
        .grid-top {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1.15fr);
            gap: 18px;
        }

        .card {
            background: white;
            border: 1px solid var(--border);
            border-radius: 20px;
            box-shadow: var(--shadow);
        }

        .card-header {
            padding: 22px 22px 0;
            display: flex;
            align-items: center;
            gap: 11px;
        }

        .card-icon {
            width: 34px;
            height: 34px;
            border-radius: 11px;
            display: grid;
            place-items: center;
            background: var(--green-100);
            color: var(--green-700);
            font-size: 16px;
        }

        .card-title {
            font-size: 15px;
            font-weight: 800;
        }

        .card-subtitle {
            padding: 8px 22px 0;
            color: var(--muted);
            font-size: 12px;
        }

        /* ---------------- Upload ---------------- */
        .upload-card {
            padding-bottom: 20px;
        }

        .drop-zone {
            margin: 20px 22px 0;
            min-height: 280px;
            border: 2px dashed #72bf88;
            border-radius: 17px;
            background: linear-gradient(180deg, #fbfefc, #f4fbf6);
            display: grid;
            place-items: center;
            text-align: center;
            padding: 28px;
            cursor: pointer;
            transition: .2s ease;
        }

        .drop-zone.dragging {
            transform: scale(.99);
            background: #e9f8ed;
            border-color: var(--green-600);
        }

        .upload-icon {
            width: 64px;
            height: 64px;
            border-radius: 20px;
            display: grid;
            place-items: center;
            background: var(--green-100);
            color: var(--green-700);
            font-size: 29px;
            margin: 0 auto 15px;
        }

        .drop-zone h3 {
            margin: 0;
            font-size: 16px;
        }

        .drop-zone p {
            margin: 8px 0 17px;
            color: var(--muted);
            font-size: 12px;
        }

        .primary-btn {
            border: 0;
            color: white;
            background: linear-gradient(135deg, #1ca552, #087338);
            border-radius: 10px;
            padding: 12px 20px;
            font-weight: 800;
            cursor: pointer;
            box-shadow: 0 9px 22px rgba(16, 139, 65, .22);
        }

        .primary-btn:disabled {
            opacity: .55;
            cursor: not-allowed;
            box-shadow: none;
        }

        .upload-meta {
            margin: 12px 22px 0;
            display: flex;
            justify-content: space-between;
            gap: 10px;
            color: var(--muted);
            font-size: 10px;
        }

        .tip {
            margin: 16px 22px 0;
            padding: 12px 14px;
            background: #f0f9f1;
            border: 1px solid #dbeedf;
            border-radius: 12px;
            color: #4b6656;
            font-size: 11px;
        }

        /* ---------------- Result ---------------- */
        .result-card {
            overflow: hidden;
        }

        .result-body {
            padding: 18px 22px 22px;
        }

        .result-layout {
            display: grid;
            grid-template-columns: 1fr .9fr;
            gap: 18px;
        }

        .preview {
            position: relative;
            overflow: hidden;
            border-radius: 16px;
            background: #e9efeb;
            min-height: 330px;
        }

        .preview img {
            width: 100%;
            height: 100%;
            min-height: 330px;
            object-fit: cover;
            display: block;
        }

        .empty-preview {
            min-height: 330px;
            display: grid;
            place-items: center;
            color: #94a59b;
            text-align: center;
            padding: 30px;
            font-size: 12px;
        }

        .result-info {
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .status {
            display: inline-flex;
            align-self: flex-start;
            gap: 7px;
            align-items: center;
            padding: 7px 10px;
            border-radius: 999px;
            font-size: 10px;
            font-weight: 800;
            background: var(--danger-bg);
            color: var(--danger);
            margin-bottom: 12px;
        }

        .status.healthy {
            background: var(--green-100);
            color: var(--green-800);
        }

        .result-info h2 {
            margin: 0;
            font-size: 28px;
            letter-spacing: -.035em;
        }

        .raw-label {
            margin-top: 5px;
            color: var(--muted);
            font-size: 11px;
        }

        .confidence-row {
            display: flex;
            justify-content: space-between;
            align-items: end;
            margin-top: 25px;
        }

        .confidence-row span:first-child {
            font-size: 12px;
            font-weight: 700;
        }

        .confidence-value {
            color: var(--green-700);
            font-size: 21px;
            font-weight: 800;
        }

        .progress {
            height: 9px;
            margin-top: 9px;
            background: #e8eeea;
            border-radius: 999px;
            overflow: hidden;
        }

        .progress > div {
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, #1b9b4d, #2fc46a);
            transition: width .5s ease;
        }

        .info-box {
            margin-top: 18px;
            padding: 13px;
            border-radius: 13px;
            background: #f0f8f1;
            color: #51665a;
            font-size: 11px;
            line-height: 1.55;
        }

        .action-row {
            display: flex;
            gap: 9px;
            margin-top: 18px;
        }

        .secondary-btn {
            flex: 1;
            border: 1px solid #b8d9c1;
            color: var(--green-800);
            background: white;
            border-radius: 10px;
            padding: 11px 12px;
            font-weight: 800;
            cursor: pointer;
        }

        .secondary-btn:hover {
            background: var(--green-50);
        }

        /* ---------------- Insight strip ---------------- */
        .insights {
            margin-top: 18px;
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1px;
            background: var(--border);
            border: 1px solid var(--border);
            border-radius: 18px;
            overflow: hidden;
            box-shadow: var(--shadow);
        }

        .insight {
            background: white;
            padding: 20px;
            display: flex;
            gap: 12px;
            align-items: flex-start;
        }

        .insight-icon {
            width: 40px;
            height: 40px;
            flex: 0 0 40px;
            border-radius: 50%;
            display: grid;
            place-items: center;
            background: var(--green-100);
            color: var(--green-700);
        }

        .insight:nth-child(2) .insight-icon {
            background: #edf4ff;
            color: #3b73c4;
        }

        .insight:nth-child(3) .insight-icon {
            background: #f8efff;
            color: #9c4cc7;
        }

        .insight:nth-child(4) .insight-icon {
            background: #fff5e5;
            color: #cf8a16;
        }

        .insight strong {
            display: block;
            font-size: 12px;
            margin-bottom: 5px;
        }

        .insight p {
            margin: 0;
            color: var(--muted);
            font-size: 10px;
            line-height: 1.55;
        }

        /* ---------------- Bottom ---------------- */
        .bottom-grid {
            margin-top: 18px;
            display: grid;
            grid-template-columns: 1.15fr 1fr .95fr;
            gap: 18px;
        }

        .bottom-card {
            padding: 20px;
        }

        .bottom-card h3 {
            margin: 0;
            font-size: 14px;
        }

        .bottom-card p {
            color: var(--muted);
            font-size: 11px;
            line-height: 1.65;
        }

        .check-list {
            margin: 14px 0 0;
            padding: 0;
            list-style: none;
            display: grid;
            gap: 9px;
        }

        .check-list li {
            font-size: 10px;
            color: #56685e;
            display: flex;
            gap: 8px;
            align-items: flex-start;
        }

        .check {
            color: var(--green-600);
            font-weight: 900;
        }

        .history {
            margin-top: 14px;
            display: grid;
            gap: 9px;
        }

        .history-item {
            display: flex;
            align-items: center;
            gap: 9px;
            padding: 9px;
            border-radius: 11px;
            background: #f7faf8;
        }

        .history-dot {
            width: 30px;
            height: 30px;
            border-radius: 9px;
            background: var(--green-100);
            display: grid;
            place-items: center;
            color: var(--green-700);
            font-size: 13px;
        }

        .history-item strong {
            display: block;
            font-size: 10px;
        }

        .history-item span {
            color: var(--muted);
            font-size: 9px;
        }

        .history-status {
            margin-left: auto;
            font-size: 8px;
            padding: 5px 7px;
            border-radius: 999px;
            background: var(--green-100);
            color: var(--green-800);
            font-weight: 800;
        }

        .history-status.disease {
            background: var(--danger-bg);
            color: var(--danger);
        }

        .footer-banner {
            margin-top: 18px;
            padding: 16px 20px;
            border: 1px solid #dcebdc;
            border-radius: 16px;
            background: linear-gradient(90deg, #eef8ef, #f9fcf8);
            color: #52655a;
            font-size: 11px;
            display: flex;
            justify-content: space-between;
            gap: 20px;
            align-items: center;
        }

        .footer-banner strong {
            color: var(--green-800);
        }

        .loading {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(4, 27, 18, .38);
            backdrop-filter: blur(5px);
            z-index: 100;
            place-items: center;
        }

        .loading.show {
            display: grid;
        }

        .loader-box {
            background: white;
            border-radius: 18px;
            padding: 28px 35px;
            text-align: center;
            box-shadow: 0 25px 70px rgba(0,0,0,.18);
        }

        .spinner {
            width: 42px;
            height: 42px;
            margin: 0 auto 13px;
            border: 4px solid #dcecdf;
            border-top-color: var(--green-600);
            border-radius: 50%;
            animation: spin .8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .toast {
            position: fixed;
            right: 22px;
            bottom: 22px;
            max-width: 360px;
            background: #13271e;
            color: white;
            padding: 13px 16px;
            border-radius: 12px;
            font-size: 11px;
            box-shadow: 0 15px 40px rgba(0,0,0,.2);
            opacity: 0;
            transform: translateY(12px);
            pointer-events: none;
            transition: .25s ease;
            z-index: 200;
        }

        .toast.show {
            opacity: 1;
            transform: translateY(0);
        }

        /* ---------------- Responsive ---------------- */
        @media (max-width: 1180px) {
            .sidebar {
                width: 230px;
                flex-basis: 230px;
            }

            .grid-top,
            .bottom-grid {
                grid-template-columns: 1fr;
            }

            .insights {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        @media (max-width: 820px) {
            .app {
                display: block;
            }

            .sidebar {
                position: static;
                width: 100%;
                height: auto;
                min-height: auto;
                padding: 16px;
            }

            .brand {
                padding-bottom: 15px;
            }

            .nav {
                display: flex;
                overflow-x: auto;
                padding-bottom: 2px;
            }

            .nav-item {
                flex: 0 0 auto;
            }

            .sidebar-card,
            .sidebar-footer {
                display: none;
            }

            .main {
                padding: 20px 14px 28px;
            }

            .topbar {
                align-items: flex-start;
            }

            .user-box {
                display: none;
            }

            .result-layout {
                grid-template-columns: 1fr;
            }

            .preview,
            .preview img,
            .empty-preview {
                min-height: 250px;
            }
        }

        @media (max-width: 560px) {
            .insights {
                grid-template-columns: 1fr;
            }

            .upload-meta {
                display: block;
                line-height: 1.6;
            }

            .action-row {
                flex-direction: column;
            }

            .drop-zone {
                min-height: 240px;
            }
        }
    </style>
</head>

<body>
<div class="app">

    <aside class="sidebar">
        <div class="brand">
            <div class="brand-icon">🌿</div>
            <div>
                <div class="brand-title">Tomato Leaf</div>
                <div class="brand-subtitle">Disease Detector</div>
            </div>
        </div>

        <nav class="nav">
            <button class="nav-item active" onclick="scrollToSection('dashboard')">
                <span class="nav-icon">⌂</span>
                <span class="nav-text">Dashboard</span>
            </button>

            <button class="nav-item" onclick="focusUpload()">
                <span class="nav-icon">⌁</span>
                <span class="nav-text">Analyze Leaf</span>
            </button>

            <button class="nav-item" onclick="showGuide()">
                <span class="nav-icon">▤</span>
                <span class="nav-text">Disease Guide</span>
            </button>

            <button class="nav-item" onclick="showGuide()">
                <span class="nav-icon">✓</span>
                <span class="nav-text">Prevention Tips</span>
            </button>

            <button class="nav-item" onclick="scrollToSection('history')">
                <span class="nav-icon">◷</span>
                <span class="nav-text">History</span>
            </button>

            <button class="nav-item" onclick="showAbout()">
                <span class="nav-icon">ⓘ</span>
                <span class="nav-text">About Project</span>
            </button>
        </nav>

        <div class="sidebar-card">
            <h4>▣ AI Powered Detection</h4>
            <p>
                Advanced deep-learning based tomato leaf classification
                using the project's trained DenseNet121 model.
            </p>
        </div>

        <div class="sidebar-footer">
            © 2026 Tomato Leaf Disease Detector<br>
            AI Powered • Smart Farming
        </div>
    </aside>

    <main class="main" id="dashboard">

        <header class="topbar">
            <div>
                <div class="eyebrow">AI Plant Health Platform</div>
                <h1>Welcome back! 👋</h1>
                <p>Detect tomato leaf diseases with the power of AI.</p>
            </div>

            <div class="user-box">
                <button class="theme" onclick="toggleTheme()" title="Toggle theme">☼</button>
                <div class="avatar">S</div>
                <div class="user-info">
                    <strong>Shihab</strong>
                    <span>User</span>
                </div>
            </div>
        </header>

        <section class="grid-top">

            <!-- Upload -->
            <div class="card upload-card" id="uploadCard">
                <div class="card-header">
                    <div class="card-icon">⇧</div>
                    <div class="card-title">Upload Tomato Leaf Image</div>
                </div>

                <div class="card-subtitle">
                    Upload a clear image of the tomato leaf to detect its condition.
                </div>

                <div
                    class="drop-zone"
                    id="dropZone"
                    onclick="document.getElementById('fileInput').click()"
                >
                    <div>
                        <div class="upload-icon">☁</div>
                        <h3>Drag & drop your leaf image here</h3>
                        <p>or choose an image from your device</p>

                        <button
                            class="primary-btn"
                            type="button"
                            onclick="event.stopPropagation(); document.getElementById('fileInput').click()"
                        >
                            Choose Image
                        </button>

                        <input
                            id="fileInput"
                            type="file"
                            accept=".jpg,.jpeg,.png,image/jpeg,image/png"
                            hidden
                        >
                    </div>
                </div>

                <div class="upload-meta">
                    <span>Supported: JPG, JPEG, PNG</span>
                    <span>Maximum size: 5 MB</span>
                </div>

                <div class="tip">
                    💡 <strong>Tip:</strong> Use a clear image with good lighting
                    and keep the leaf visible.
                </div>
            </div>

            <!-- Result -->
            <div class="card result-card">
                <div class="card-header">
                    <div class="card-icon">⌁</div>
                    <div class="card-title">Prediction Result</div>
                </div>

                <div class="result-body">
                    <div class="result-layout">

                        <div class="preview" id="preview">
                            <div class="empty-preview" id="emptyPreview">
                                Your analyzed leaf image will appear here.
                            </div>
                        </div>

                        <div class="result-info" id="resultInfo">
                            <div class="status" id="status">
                                <span>●</span>
                                <span>Waiting for image</span>
                            </div>

                            <h2 id="resultLabel">No prediction yet</h2>
                            <div class="raw-label" id="rawLabel">
                                Upload an image to start analysis.
                            </div>

                            <div class="confidence-row">
                                <span>Confidence Score</span>
                                <span class="confidence-value" id="confidence">—</span>
                            </div>

                            <div class="progress">
                                <div id="progressBar" style="width:0%"></div>
                            </div>

                            <div class="info-box" id="resultMessage">
                                The AI model will analyze the uploaded leaf using
                                the same 224×224 preprocessing pipeline used in
                                the project.
                            </div>

                            <div class="action-row">
                                <button class="secondary-btn" onclick="focusUpload()">
                                    ↻ Analyze Another
                                </button>
                                <button class="secondary-btn" onclick="showTop3()">
                                    View Top-3
                                </button>
                            </div>
                        </div>

                    </div>
                </div>
            </div>
        </section>

        <!-- Insights -->
        <section class="insights">
            <div class="insight">
                <div class="insight-icon">✓</div>
                <div>
                    <strong>Early Detection</strong>
                    <p>Identify possible leaf conditions before taking the next step.</p>
                </div>
            </div>

            <div class="insight">
                <div class="insight-icon">◎</div>
                <div>
                    <strong>Better Yield</strong>
                    <p>Use timely information to support smarter crop management.</p>
                </div>
            </div>

            <div class="insight">
                <div class="insight-icon">◇</div>
                <div>
                    <strong>Save Resources</strong>
                    <p>Make informed decisions and avoid unnecessary interventions.</p>
                </div>
            </div>

            <div class="insight">
                <div class="insight-icon">▥</div>
                <div>
                    <strong>Smart Farming</strong>
                    <p>Bring AI-based image classification into a simple workflow.</p>
                </div>
            </div>
        </section>

        <!-- Bottom cards -->
        <section class="bottom-grid">

            <div class="card bottom-card">
                <h3>ⓘ About the AI Model</h3>
                <p>
                    This interface uses the trained model supplied with the project.
                    The notebook defines a DenseNet121-based architecture with
                    global average/max pooling and a multi-layer classification head.
                </p>

                <ul class="check-list">
                    <li><span class="check">●</span> 224 × 224 RGB input</li>
                    <li><span class="check">●</span> CLAHE preprocessing</li>
                    <li><span class="check">●</span> 9 output classes</li>
                    <li><span class="check">●</span> Softmax probability output</li>
                </ul>
            </div>

            <div class="card bottom-card">
                <h3>✓ Recommended Workflow</h3>
                <p>
                    Use the prediction as an AI-assisted screening result and
                    verify important agricultural decisions with appropriate
                    expert guidance.
                </p>

                <ul class="check-list">
                    <li><span class="check">✓</span> Capture a clear leaf image</li>
                    <li><span class="check">✓</span> Upload and analyze</li>
                    <li><span class="check">✓</span> Review confidence and top predictions</li>
                    <li><span class="check">✓</span> Take the appropriate next step</li>
                </ul>
            </div>

            <div class="card bottom-card" id="history">
                <h3>◷ Recent History</h3>
                <p>Predictions made during this browser session.</p>

                <div class="history" id="historyList">
                    <div class="history-item">
                        <div class="history-dot">⌁</div>
                        <div>
                            <strong>No scans yet</strong>
                            <span>Upload your first leaf</span>
                        </div>
                    </div>
                </div>
            </div>

        </section>

        <div class="footer-banner">
            <span>🌱 <strong>Protect your crops</strong> with AI-assisted plant health screening.</span>
            <strong>Smart Farming, Better Future.</strong>
        </div>

    </main>
</div>

<div class="loading" id="loading">
    <div class="loader-box">
        <div class="spinner"></div>
        <strong>Analyzing leaf...</strong>
        <div style="margin-top:6px;color:#708078;font-size:11px;">
            Running the trained DenseNet121 model
        </div>
    </div>
</div>

<div class="toast" id="toast"></div>

<script>
    const fileInput = document.getElementById("fileInput");
    const dropZone = document.getElementById("dropZone");
    const loading = document.getElementById("loading");

    let latestPrediction = null;
    let historyItems = [];

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length) {
            analyzeFile(fileInput.files[0]);
        }
    });

    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, e => {
            e.preventDefault();
            dropZone.classList.add("dragging");
        });
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, e => {
            e.preventDefault();
            dropZone.classList.remove("dragging");
        });
    });

    dropZone.addEventListener("drop", e => {
        const files = e.dataTransfer.files;
        if (files.length) analyzeFile(files[0]);
    });

    async function analyzeFile(file) {
        const allowed = ["image/jpeg", "image/png"];
        if (!allowed.includes(file.type)) {
            showToast("Please upload a JPG, JPEG or PNG image.");
            return;
        }

        if (file.size > 5 * 1024 * 1024) {
            showToast("Image is larger than the 5 MB limit.");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        loading.classList.add("show");

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Prediction failed.");
            }

            latestPrediction = data;
            updateResult(data, file.name);
            addHistory(data, file.name);

        } catch (error) {
            showToast(error.message);
        } finally {
            loading.classList.remove("show");
        }
    }

    function updateResult(data, filename) {
        document.getElementById("preview").innerHTML =
            `<img src="${data.image}" alt="Uploaded tomato leaf">`;

        const healthy = data.raw_label === "healthy";
        const status = document.getElementById("status");

        status.className = "status" + (healthy ? " healthy" : "");
        status.innerHTML = healthy
            ? "<span>●</span><span>Healthy Leaf</span>"
            : "<span>●</span><span>Condition Detected</span>";

        document.getElementById("resultLabel").textContent = data.label;
        document.getElementById("rawLabel").textContent =
            filename + " • " + data.raw_label;

        document.getElementById("confidence").textContent =
            data.confidence.toFixed(2) + "%";

        document.getElementById("progressBar").style.width =
            Math.min(data.confidence, 100) + "%";

        document.getElementById("resultMessage").textContent =
            healthy
            ? "The model classified this image as a healthy tomato leaf."
            : "The model detected a condition from the project's trained 9-class label set. Review the confidence and top predictions before making decisions.";

        document.getElementById("resultInfo").scrollIntoView({
            behavior: "smooth",
            block: "nearest"
        });
    }

    function addHistory(data, filename) {
        historyItems.unshift({
            label: data.label,
            confidence: data.confidence,
            healthy: data.raw_label === "healthy",
            filename
        });

        historyItems = historyItems.slice(0, 4);

        const container = document.getElementById("historyList");
        container.innerHTML = historyItems.map(item => `
            <div class="history-item">
                <div class="history-dot">${item.healthy ? "✓" : "!"}</div>
                <div>
                    <strong>${escapeHtml(item.label)}</strong>
                    <span>${item.confidence.toFixed(2)}% • ${escapeHtml(item.filename)}</span>
                </div>
                <div class="history-status ${item.healthy ? "" : "disease"}">
                    ${item.healthy ? "Healthy" : "Detected"}
                </div>
            </div>
        `).join("");
    }

    function showTop3() {
        if (!latestPrediction) {
            showToast("Analyze an image first.");
            return;
        }

        const text = latestPrediction.top3
            .map((x, i) => `${i + 1}. ${x.label} — ${x.confidence.toFixed(2)}%`)
            .join("\n");

        alert("Top-3 Predictions\n\n" + text);
    }

    function focusUpload() {
        document.getElementById("uploadCard").scrollIntoView({
            behavior: "smooth",
            block: "center"
        });
        setTimeout(() => fileInput.click(), 350);
    }

    function scrollToSection(id) {
        document.getElementById(id).scrollIntoView({
            behavior: "smooth",
            block: "start"
        });
    }

    function showGuide() {
        alert(
            "Disease Guide\n\n" +
            "The trained model supports 9 classes:\n\n" +
            "• Bacterial Spot\n" +
            "• Early Blight\n" +
            "• Late Blight\n" +
            "• Leaf Miner\n" +
            "• Leaf Mold\n" +
            "• Septoria Leaf Spot\n" +
            "• Spider Mites\n" +
            "• Verticillium Wilt\n" +
            "• Healthy Leaf"
        );
    }

    function showAbout() {
        alert(
            "Tomato Leaf Disease Detector\n\n" +
            "AI-assisted image classification project using the supplied trained model.\n\n" +
            "Input: 224 × 224 RGB\n" +
            "Architecture: DenseNet121-based\n" +
            "Classes: 9"
        );
    }

    function toggleTheme() {
        document.body.classList.toggle("dark-preview");

        if (document.body.classList.contains("dark-preview")) {
            document.body.style.filter = "brightness(.88)";
            showToast("Preview mode enabled.");
        } else {
            document.body.style.filter = "";
            showToast("Light mode restored.");
        }
    }

    function showToast(message) {
        const toast = document.getElementById("toast");
        toast.textContent = message;
        toast.classList.add("show");

        clearTimeout(window.toastTimer);
        window.toastTimer = setTimeout(() => {
            toast.classList.remove("show");
        }, 2800);
    }

    function escapeHtml(value) {
        return String(value)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
    }
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image file was uploaded."}), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({"error": "Please select an image."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only JPG, JPEG and PNG images are supported."}), 400

    image_bytes = file.read()

    if not image_bytes:
        return jsonify({"error": "The uploaded image is empty."}), 400

    if len(image_bytes) > MAX_FILE_SIZE:
        return jsonify({"error": "Maximum image size is 5 MB."}), 400

    try:
        result = predict_image(image_bytes)
        return jsonify(result)

    except Exception as exc:
        print("Prediction error:", repr(exc))
        return jsonify({
            "error": "Could not analyze this image. Please try another clear leaf image."
        }), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "Maximum image size is 5 MB."}), 413


if __name__ == "__main__":
    # Development server.
    # Open http://127.0.0.1:5000 in your browser.
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
