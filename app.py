
import os
import gradio as gr

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------

APP_TITLE = "TomatoCare AI"
MODEL_ARCH = "DenseNet121"
VAL_ACCURACY = "98.5%"
PRECISION_SCORE = "97.2%"

# -----------------------------------------------------------------------------
# Gradio callback stubs (Replace with your actual model inference)
# -----------------------------------------------------------------------------

def run_diagnosis(image):
    if image is None:
        raise gr.Error("Please upload a tomato leaf image first.")
    
    # Placeholder for actual model inference
    # result = predict_leaf_disease(image)
    
    # Mock result for UI demonstration
    mock_prediction = "Healthy" 
    mock_confidence = 99.8
    
    result_markdown = (
        f"## Diagnosis: **{mock_prediction}**\n\n"
        f"**Confidence:** {mock_confidence:.1f}%\n\n"
        "> Disclaimer: This is an AI prediction. Please consult a local agricultural expert for definitive action."
    )
    
    disease_info = "The uploaded leaf shows no visible signs of major diseases such as Late Blight, Early Blight, or Septoria Leaf Spot."
    
    treatment = "Continue current watering and fertilizing schedule. Ensure adequate sunlight and air circulation."

    return result_markdown, disease_info, treatment

def clear_outputs():
    return (
        None,
        "## Ready for Diagnosis\n\nUpload a tomato leaf image and click **Analyze Leaf**.",
        "",
        ""
    )

# -----------------------------------------------------------------------------
# Custom CSS for TomatoCare AI
# -----------------------------------------------------------------------------

CUSTOM_CSS = """
:root {
    --tc-bg-dark: #0f1715;     /* Very dark green-tinted black */
    --tc-bg-card: #15221d;     /* Slightly lighter dark green for cards */
    --tc-primary: #34d399;     /* Mint green for primary buttons/accents */
    --tc-primary-hover: #10b981; 
    --tc-text-main: #f8fafc;
    --tc-text-muted: #94a3b8;
    --tc-border: rgba(52, 211, 153, 0.2);
}

body, .gradio-container {
    background: var(--tc-bg-dark) !important;
    color: var(--tc-text-main) !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}

/* Reset default gradio padding/margin for custom layout */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 0 !important;
}

/* Navbar */
.tc-navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 40px;
    background: rgba(15, 23, 21, 0.8);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--tc-border);
    position: sticky;
    top: 0;
    z-index: 100;
}

.tc-brand {
    font-size: 24px;
    font-weight: 800;
    color: var(--tc-primary);
    display: flex;
    align-items: center;
    gap: 10px;
}

.tc-nav-links {
    display: flex;
    gap: 30px;
}

.tc-nav-links a {
    color: var(--tc-text-main);
    text-decoration: none;
    font-weight: 500;
    font-size: 15px;
    transition: color 0.2s;
}

.tc-nav-links a:hover {
    color: var(--tc-primary);
}

.tc-btn-main {
    background: var(--tc-primary);
    color: #022c22; /* Dark text on mint bg */
    padding: 10px 24px;
    border-radius: 8px;
    font-weight: 700;
    text-decoration: none;
    transition: background 0.2s;
}

.tc-btn-main:hover {
    background: var(--tc-primary-hover);
}

/* Hero Section */
.tc-hero {
    display: flex;
    align-items: center;
    padding: 80px 40px;
    gap: 60px;
}

.tc-hero-text {
    flex: 1;
}

.tc-hero-title {
    font-size: 56px;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 20px;
}

.tc-hero-title span {
    color: var(--tc-primary);
}

.tc-hero-subtitle {
    font-size: 18px;
    color: var(--tc-text-muted);
    line-height: 1.6;
    margin-bottom: 40px;
    max-width: 600px;
}

/* Stats Row */
.tc-stats-container {
    display: flex;
    gap: 20px;
    margin-top: 40px;
}

.tc-stat-box {
    background: var(--tc-bg-card);
    border: 1px solid var(--tc-border);
    border-radius: 12px;
    padding: 20px;
    flex: 1;
}

.tc-stat-value {
    font-size: 28px;
    font-weight: 700;
    color: var(--tc-text-main);
    margin-bottom: 5px;
}

.tc-stat-label {
    font-size: 14px;
    color: var(--tc-text-muted);
}

/* Upload Section */
.tc-upload-section {
    flex: 0.8;
    background: var(--tc-bg-card);
    border: 1px solid var(--tc-border);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.4);
}

.tc-upload-title {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 20px;
}

/* Features Section */
.tc-features {
    padding: 60px 40px;
    text-align: center;
}

.tc-features h2 {
    font-size: 36px;
    margin-bottom: 40px;
}

.tc-feature-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 24px;
}

.tc-feature-card {
    background: var(--tc-bg-card);
    border: 1px solid var(--tc-border);
    border-radius: 16px;
    padding: 30px 20px;
    text-align: left;
}

.tc-feature-icon {
    font-size: 32px;
    color: var(--tc-primary);
    margin-bottom: 20px;
}

.tc-feature-card h3 {
    font-size: 18px;
    margin-bottom: 10px;
}

.tc-feature-card p {
    color: var(--tc-text-muted);
    font-size: 14px;
    line-height: 1.5;
}

/* Gradio Overrides */
#image-upload {
    border: 2px dashed rgba(52, 211, 153, 0.4) !important;
    background: rgba(0,0,0,0.2) !important;
    border-radius: 12px !important;
    min-height: 250px !important;
}

#analyze-btn {
    background: var(--tc-primary) !important;
    color: #022c22 !important;
    border: none !important;
    font-weight: bold !important;
    margin-top: 15px !important;
}

#analyze-btn:hover {
    background: var(--tc-primary-hover) !important;
}

#clear-btn {
    background: transparent !important;
    border: 1px solid var(--tc-text-muted) !important;
    color: var(--tc-text-main) !important;
    margin-top: 15px !important;
}

.tc-output-card {
    background: var(--tc-bg-card) !important;
    border: 1px solid var(--tc-border) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    margin-top: 20px !important;
}

/* Footer */
.tc-footer {
    text-align: center;
    padding: 40px;
    color: var(--tc-text-muted);
    font-size: 14px;
    border-top: 1px solid var(--tc-border);
    margin-top: 40px;
}

@media (max-width: 900px) {
    .tc-hero { flex-direction: column; padding: 40px 20px; }
    .tc-feature-grid { grid-template-columns: repeat(2, 1fr); }
    .tc-stats-container { flex-wrap: wrap; }
    .tc-nav-links { display: none; }
}
@media (max-width: 600px) {
    .tc-feature-grid { grid-template-columns: 1fr; }
    .tc-hero-title { font-size: 40px; }
}
"""

theme = gr.themes.Monochrome(
    primary_hue="emerald",
    neutral_hue="slate",
).set(
    body_background_fill="#0f1715",
    block_background_fill="#15221d",
    block_border_color="rgba(52, 211, 153, 0.2)",
    button_primary_background_fill="#34d399",
    button_primary_text_color="#022c22",
)

# -----------------------------------------------------------------------------
# UI Layout
# -----------------------------------------------------------------------------

with gr.Blocks(title=APP_TITLE, theme=theme, css=CUSTOM_CSS) as demo:
    
    # Navbar
    gr.HTML("""
        <div class="tc-navbar">
            <div class="tc-brand">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
                TomatoCare AI
            </div>
            <div class="tc-nav-links">
                <a href="#">Home</a>
                <a href="#">About</a>
                <a href="#">Model Architecture</a>
                <a href="#">Dataset</a>
                <a href="#">Contact</a>
            </div>
        </div>
    """)

    # Main Hero & Upload Area
    with gr.Row(elem_classes="tc-hero"):
        
        # Left Content: Text & Stats
        with gr.Column(scale=1, elem_classes="tc-hero-text"):
            gr.HTML(f"""
                <h1 class="tc-hero-title">Diagnose <span>Tomato Diseases</span>.<br>Protect Your Harvest.</h1>
                <p class="tc-hero-subtitle">
                    Leverage our fine-tuned {MODEL_ARCH} deep learning model to instantly analyze tomato leaf images and accurately identify over 9 specific plant diseases, helping you take targeted action to save your crops.
                </p>
            """)
            
            gr.HTML(f"""
                <div class="tc-stats-container">
                    <div class="tc-stat-box">
                        <div class="tc-stat-value">12,000+</div>
                        <div class="tc-stat-label">Leaf Images Trained</div>
                    </div>
                    <div class="tc-stat-box">
                        <div class="tc-stat-value">{VAL_ACCURACY}</div>
                        <div class="tc-stat-label">Validation Accuracy</div>
                    </div>
                    <div class="tc-stat-box">
                        <div class="tc-stat-value">{PRECISION_SCORE}</div>
                        <div class="tc-stat-label">Precision Score</div>
                    </div>
                </div>
            """)
            
        # Right Content: Upload & Result interface
        with gr.Column(scale=1, elem_classes="tc-upload-section"):
            gr.HTML('<div class="tc-upload-title">Scan a Tomato Leaf</div>')
            
            image_input = gr.Image(
                type="pil", 
                label="Upload Image", 
                show_label=False,
                elem_id="image-upload"
            )
            
            with gr.Row():
                analyze_btn = gr.Button("Analyze Leaf", elem_id="analyze-btn")
                clear_btn = gr.Button("Clear", elem_id="clear-btn")
                
            result_display = gr.Markdown(
                "## Ready for Diagnosis\n\nUpload a tomato leaf image and click **Analyze Leaf**.",
                elem_classes="tc-output-card"
            )
            
            with gr.Accordion("Detailed Analysis", open=False):
                disease_info = gr.Textbox(label="Symptom Details", interactive=False)
                treatment_info = gr.Textbox(label="Recommended Action", interactive=False)

    # Features Section
    gr.HTML("""
        <div class="tc-features">
            <h2>Why Choose TomatoCare AI?</h2>
            <div class="tc-feature-grid">
                <div class="tc-feature-card">
                    <div class="tc-feature-icon">🔍</div>
                    <h3>Detailed Symptom Identification</h3>
                    <p>Analyzes leaf textures, discoloration, and lesion patterns to detect early signs of specific pathogens.</p>
                </div>
                <div class="tc-feature-card">
                    <div class="tc-feature-icon">🌿</div>
                    <h3>Health Status Interpretation</h3>
                    <p>Categorizes leaves into distinct disease classes or confirms a healthy status with high confidence.</p>
                </div>
                <div class="tc-feature-card">
                    <div class="tc-feature-icon">⚙️</div>
                    <h3>Dense Feature Optimization</h3>
                    <p>Utilizes the DenseNet121 architecture for efficient feature reuse, maximizing prediction accuracy.</p>
                </div>
                <div class="tc-feature-card">
                    <div class="tc-feature-icon">🎯</div>
                    <h3>Targeted Disease Diagnosis</h3>
                    <p>Differentiates between visually similar issues like Bacterial Spot, Early Blight, and Septoria Leaf Spot.</p>
                </div>
            </div>
        </div>
    """)
    
    # Footer
    gr.HTML("""
        <div class="tc-footer">
            <p><strong>TomatoCare AI</strong> • Powered by DenseNet121 & TensorFlow</p>
            <p style="margin-top: 10px;">Built by Farhan Sadik Shihab & Team</p>
        </div>
    """)

    # Event Listeners
    analyze_btn.click(
        fn=run_diagnosis,
        inputs=[image_input],
        outputs=[result_display, disease_info, treatment_info]
    )
    
    clear_btn.click(
        fn=clear_outputs,
        inputs=[],
        outputs=[image_input, result_display, disease_info, treatment_info]
    )

if __name__ == "__main__":
    demo.launch()
