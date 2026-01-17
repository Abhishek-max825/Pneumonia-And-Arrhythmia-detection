# NeuroX: AI-Powered Multi-Modal Medical Diagnostics
![Version](https://img.shields.io/badge/version-1.0.0-blue.svg) ![Framework](https://img.shields.io/badge/framework-PyTorch%20%7C%20Streamlit-red.svg) ![License](https://img.shields.io/badge/license-MIT-green.svg)

**NeuroX** is a research-grade, offline-capable medical imaging and signal processing application designed to assist clinicians in the early detection of **Pneumonia** (via Chest X-Rays) and **Cardiac Arrhythmias** (via 12-Lead ECGs).

By bridging the gap between "Black Box" Deep Learning and clinical trust, NeuroX features **Explainable AI (XAI)** to visualize exactly *why* a diagnosis was made.

---

## 🌟 Key Features

### 1. Advanced Chest X-Ray Analysis (Vision)
*   **Model**: Fine-tuned `EfficientNet-B0` / `MobileNetV2` (Transfer Learning).
*   **Performance**: Optimized for high sensitivity in detecting lung opacities.
*   **Explainability**: Integrated **Grad-CAM (Gradient-weighted Class Activation Mapping)** generates a heatmap overlay, pinpointing the exact lung regions contributing to the "Pneumonia" prediction.
*   **Smart Contrast**: Built-in CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing for enhanced visibility of bone structures and soft tissues.

### 2. Clinical-Grade ECG Interpretation (Time-Series)
*   **Model**: Custom `ResNet1D-18` (Deep Residual CNN optimized for 1D signals).
*   **Input Standard**: Supports standard 12-Lead ECG data (PTB-XL / WFDB format).
*   **Diagnostics**: Classifies rhythms into **Normal Sinus Rhythm** or **Arrhythmia/Abnormality** with high confidence.
*   **Visualization**: Interactive Matplotlib rendering of Lead II rhythm strips.

### 3. Patient History Dashboard
*   **Local Database**: Built-in SQLite database automatically isolates and persists patient records—no internet required.
*   **Analytics**: Real-time dashboard showing disease prevalence rates, total scan counts, and recent diagnostic history.
*   **Audit Trail**: Every diagnosis is timestamped and logged for review.

### 4. Professional Reporting
*   **PDF Generation**: One-click generation of comprehensive medical reports containing:
    *   Patient ID & Timestamp
    *   Original Scan + AI Heatmap (side-by-side)
    *   Diagnostic Confidence & Conclusion
    *   Physician Notes Section

---

## 🛠 System Architecture

NeuroX runs entirely locally (Edge AI), prioritizing patient data privacy.

```mermaid
graph TD
    User[Clinician] -->|Uploads X-Ray/ECG| App(Streamlit Interface)
    App -->|Preprocessing| Pre{Modality?}
    
    Pre -->|Image (CLAHE)| CNN[EfficientNet-B0]
    Pre -->|Signal (Denoising)| RNN[ResNet1D-18]
    
    CNN -->|Feature Maps| GradCAM[XAI Engine]
    CNN -->|Logits| Pred1[Pneumonia Probability]
    
    RNN -->|Features| Pred2[Arrhythmia Probability]
    
    GradCAM --> Viz[Heatmap Overlay]
    
    Pred1 --> DB[(SQLite History)]
    Pred2 --> DB
    
    Viz --> UI[Dashboard Display]
```

## 🚀 Installation & Setup

### Prerequisites
*   **OS**: Windows, macOS, or Linux
*   **Python**: 3.9+
*   **Hardware**: CDUA-capable GPU recommended for training (CPU supported for inference).

### Quick Start
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-repo/neuox-diagnostics.git
    cd Pneumonia_and_Heart_Disease_Detection
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**:
    ```bash
    streamlit run app.py
    ```

## 📂 Project Structure

| Directory | Description |
| :--- | :--- |
| `src/models/` | PyTorch definitions for `EfficientNet` (XRay) and `ResNet1D` (ECG). |
| `src/utils/` | Helper modules for `db` (SQLite), `explainability` (GradCAM), and `preprocessing`. |
| `data/` | Raw storage for X-Ray images and PTB-XL ECG waveforms. |
| `ecg_testing/` | Auto-generated sample data for rapid testing. |
| `app.py` | Main entry point for the Streamlit web interface. |

## 🛡️ Medical Disclaimer
> **IMPORTANT**: This software is for **Research and Educational Purposes Only**. It is NOT approved by the FDA or any medical regulatory body for clinical use. It should not be used as the sole basis for medical diagnoses.

---
**Developed by Abhishek** | Powered by PyTorch & Streamlit
