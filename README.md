# Diabetic Retinopathy Detection & Clinical Explainability (Grad-CAM)

An automated binary screening and explainability platform for Diabetic Retinopathy (DR) using a hierarchical **Swin Transformer** (swin_base_patch4_window7_224) and **Grad-CAM** visual interpretability.

---

## Key Features

- **Hierarchical Swin Transformer**: Employs shifted-window self-attention to model both fine-grained retinal lesions (microaneurysms, hemorrhages) and global fundus context.
- **Contrast-Enhanced Preprocessing**: Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L-channel in the CIE LAB color space to amplify micro-vascular lesions without distorting natural coloration.
- **Explainable AI (Grad-CAM)**: Generates attention heatmaps and overlays showing the exact retinal regions influencing the classification decision.
- **Interactive Web Application**: Clean, binary triage dashboard built with Streamlit featuring built-in demo samples for instant evaluation.
- **Clinical Screening PDF Export**: Generates a standardized, high-resolution diagnostic screening report with side-by-side tri-panel figures, risk stratification, and patient details.

---

## Project Structure

`	ext
├── app.py                  # Streamlit web application & report generator
├── demo_samples/           # Demo fundus images for instant evaluation
│   ├── healthy_retina.png
│   └── diabetic_retinopathy.png
├── src/
│   ├── model.py            # Swin Transformer model definition
│   ├── dataset.py          # PyTorch dataset & CLAHE preprocessing
│   ├── explain.py          # Standalone Grad-CAM visualizer
│   └── train.py            # Training & validation loop
├── .streamlit/
│   └── config.toml         # Streamlit server configuration
├── requirements.txt        # Python package dependencies
└── README.md
`

---

## Quickstart (Running Locally)

### 1. Clone the Repository
`ash
git clone https://github.com/rohithv080/Diabetic-Retinopathy-Detection.git
cd Diabetic-Retinopathy-Detection
`

### 2. Install Dependencies
`ash
pip install -r requirements.txt
`

### 3. Model Weights
Place the trained model checkpoint (est_swin_model.pth) inside the src/ directory:
`	ext
src/best_swin_model.pth
`

### 4. Launch the Web Application
`ash
streamlit run app.py
`

The app will be available at http://localhost:8501.

---

## Deployment Options

### Option A: Hugging Face Spaces (Recommended for Jury & Demonstrations)
1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Choose **Streamlit** as the Space SDK and select **CPU basic** (16 GB RAM is included free).
3. Connect your GitHub repository (
ohithv080/Diabetic-Retinopathy-Detection).
4. Upload src/best_swin_model.pth directly via Git LFS or via the Hugging Face web interface into the src/ folder.

### Option B: Streamlit Community Cloud
1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Select **New App**, choose repository 
ohithv080/Diabetic-Retinopathy-Detection, branch main, and main file pp.py.
3. Set the MODEL_WEIGHTS_URL secret/environment variable pointing to your hosted model weights if needed.

---

## License
MIT License
