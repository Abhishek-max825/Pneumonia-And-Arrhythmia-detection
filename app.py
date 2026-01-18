import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from fpdf import FPDF
import base64
import io

# Import project modules
from src.models.train_xray import build_model as build_xray_model
from src.models.train_ecg import ECGNet
from src.utils.explainability import GradCAM, overlay_cam
from src.utils.db import init_db, add_record, get_history, get_stats

# --- Config ---
st.set_page_config(
    page_title="NeuroX | AI Medical Diagnostics",
    page_icon="none",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Database
init_db()

# --- Custom CSS (Dark Theme & Premium UI) ---
st.markdown("""
<style>
    /* Reduce Top Padding - Balanced */
    .block-container {
        padding-top: 3rem !important; /* Increase padding to prevent header overlap */
        padding-bottom: 0rem !important;
        margin-top: 0px; 
    }
    
    /* Hide Default Footer (Keep Header visible for Sidebar toggle) */
    footer { visibility: hidden; }
    
    /* Transparent Header Background */
    [data-testid="stHeader"] {
        background: transparent;
    }

    /* Compact Navigation */
    div[data-testid="stRadio"] {
        margin-top: 10px; /* Add space between Title and Nav */
        margin-bottom: 0px !important; /* Make tabs touch the line below */
        padding-bottom: 0px !important;
    }

    /* Zero-Gap Horizontal Rule - Clean Separation */
    hr {
        margin-top: -34px !important; /* Force pull up to touch tabs */
        margin-bottom: 20px;
        border-color: #30363D;
        border-width: 1px;
        position: relative;
        z-index: 0; /* Ensure it stays behind the active tab border */
    }

    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }

    /* Animations */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    /* Animation triggered on block container */
    .block-container {
        animation: fadeIn 0.5s ease-out;
    }
    .metric-card {
        animation: fadeIn 0.6s ease-out;
    }

    /* Navigation - Hide Default Radio Circles & Style as Pills */
    [data-testid="stRadio"] > div {
        display: flex;
        justify-content: center;
        gap: 0px; /* Zero gap between items */
        width: 100%;
        background: transparent;
    }
    [data-testid="stRadio"] label {
        background-color: transparent;
        padding: 10px 20px;
        border-radius: 0px;
        border: none;
        border-bottom: 2px solid transparent;
        cursor: pointer;
        transition: all 0.2s ease;
        text-align: center;
        font-weight: 600;
        margin-bottom: 0px !important;
        position: relative;
        overflow: hidden;
        color: #8B949E;
    }
    
    /* Hover State */
    [data-testid="stRadio"] label:hover {
        color: #00C9FF;
        border-bottom: 2px solid #00C9FF;
        background-color: #161B22; /* Subtle background on hover */
    }
    
    /* Active State Override */
    div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #E6EDF3;
        font-weight: 700;
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-top: 10px;
        padding-bottom: 5px !important;
    }

    /* Cards/Containers */
    div[data-testid="stVerticalBlock"] > div {
        /* Generic block styling if needed */
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        background: linear-gradient(90deg, #238636 0%, #2EA043 100%);
        border: none;
        color: white;
        transition: all 0.3s ease;
        height: 45px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(46, 160, 67, 0.4);
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        background-color: #161B22;
        border: 1px dashed #30363D;
        border-radius: 12px;
        padding: 20px;
        transition: border 0.3s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #00C9FF;
    }

    /* Custom Classes */
    .metric-card {
        background-color: #21262D;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #00C9FF;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #FFFFFF;
    }
    .metric-label {
        font-size: 14px;
        color: #8B949E;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar (System Status Only) ---
with st.sidebar:
    st.title("System Status")
    device = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    st.markdown(f"""
    <div style='background-color: #21262D; padding: 10px; border-radius: 8px;'>
        <small style='color: #8B949E;'>Device</small><br>
        <strong style='color: #00C9FF;'>{device}</strong>
    </div>
    """, unsafe_allow_html=True)
    if torch.cuda.is_available():
         st.caption(f"GPU: {torch.cuda.get_device_name(0)}")
         
    st.markdown("---")
    st.markdown("### 📜 Recent Diagnostics")
    
    if 'history' not in st.session_state:
        # Load from DB on first load
        try:
            df_hist = get_history()
            # Convert DataFrame to list of dicts for the UI loop
            # Expected keys: type, result, conf, color, timestamp
            history_list = []
            for _, row in df_hist.iterrows():
                # Map DB columns to UI keys
                # DB: timestamp, modality, filename, result, confidence, details
                color = "#2EA043" if row['result'] in ["NORMAL", "Normal Rhythm"] else "#FF7B72"
                history_list.append({
                    "type": row['modality'],
                    "result": row['result'],
                    "conf": f"{row['confidence']*100:.1f}%",
                    "color": color,
                    "timestamp": row['timestamp']
                })
            # Reverse to show newest first (though DB query is already DESC, let's keep consistency)
            # Actually DB query strings are DESC. df iterrows goes top to bottom (newest first).
            # The UI code iterates `reversed(st.session_state['history'][-5:])`.
            # If we prepend to list during session, list has [oldest, ..., newest].
            # So `get_history` (DESC) returns [newest, ..., oldest].
            # We need to store in valid order.
            # Let's just store it as we want.
            st.session_state['history'] = history_list[::-1] # Store oldest -> newest so the UI logic works
        except Exception as e:
            st.error(f"Failed to load history: {e}")
            st.session_state['history'] = []
    
    if not st.session_state['history']:
        st.caption("No recent analysis.")
    else:
        for item in reversed(st.session_state['history'][-5:]):
            st.markdown(f"""
            <div style='background-color: #21262D; padding: 8px; border-radius: 6px; margin-bottom: 6px; border-left: 3px solid {item.get('color', '#8B949E')};'>
                <small style='color: #E6EDF3;'>{item.get('type', 'Unknown')}</small><br>
                <div style='display: flex; justify-content: space-between;'>
                    <strong style='color: #FFFFFF; font-size: 13px;'>{item.get('result', 'Pending')}</strong>
                    <span style='color: #8B949E; font-size: 11px;'>{item.get('conf', '')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
         
    st.markdown("---")
    st.markdown("© 2026 NeuroX")

# --- Top Navigation ---
st.markdown("<h1 style='text-align: center; margin-bottom: 10px;'>NeuroX Diagnostics</h1>", unsafe_allow_html=True)

# Navigation as Horizontal Tabs
page = st.radio("Navigation", ["Home", "Pneumonia Detection", "Arrhythmia Detection", "Dashboard", "About"], 
                horizontal=True, 
                label_visibility="collapsed")

# --- Page Transition Logic ---
if 'last_page' not in st.session_state:
    st.session_state['last_page'] = page

if st.session_state['last_page'] != page:
    # Page Changed: Reset Analysis State
    st.session_state['analyze_xray'] = False
    st.session_state['analyze_ecg'] = False
    st.session_state['last_page'] = page
    
    # Clear Matplotlib Memory Aggressively
    plt.close('all')
    st.rerun()

# Dynamic CSS to force animation replay on page change
# Dynamic CSS to force animation replay on page change
st.markdown("""
    <style>
    @keyframes fadeInKEY {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    /* Target only content divs that are siblings AFTER the navigation menu */
    .block-container div:has(div[data-testid="stRadio"]) ~ div {
        animation: fadeInKEY 0.5s ease-out forwards;
    }
    </style>
""".replace("KEY", page.replace(' ', '')), unsafe_allow_html=True)
    
st.markdown("---")

# --- Helper Functions ---
@st.cache_resource(show_spinner=False)
def load_xray_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_xray_model(device)
    try:
        model.load_state_dict(torch.load('models/xray_model.pt', map_location=device))
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error("Model not found. Please train it first.")
        return None, None

@st.cache_resource(show_spinner=False)
def load_ecg_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGNet().to(device)
    try:
        model.load_state_dict(torch.load('models/ecg_model.pt', map_location=device))
        model.eval()
        return model, device
    except FileNotFoundError:
        st.error("Model not found. Please train it first.")
        return None, None

def preprocess_image(image):
    # Apply CLAHE to match training pipeline
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced_img = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(enhanced_img).unsqueeze(0)

def analyze_rr_intervals(signal):
    """
    Clinical RR interval analysis for arrhythmia detection.
    
    Args:
        signal: ECG signal array (12, 1000) or (1000,)
        
    Returns:
        dict: {
            'has_arrhythmia': bool,
            'rr_coefficient_variation': float,
            'mean_rr': float,
            'confidence': float
        }
    """
    from scipy.signal import find_peaks, butter, filtfilt
    
    # Use Lead II (index 1) or first available lead
    if signal.ndim == 2:
        lead = signal[1] if signal.shape[0] > 1 else signal[0]
    else:
        lead = signal.flatten()
    
    # Step 1: Baseline wander removal using high-pass filter
    try:
        # Remove baseline wander (< 0.5 Hz)
        # PTB-XL is sampled at 100 Hz
        fs = 100  # Sampling frequency in Hz
        nyquist = fs / 2
        cutoff = 0.5
        b, a = butter(3, cutoff / nyquist, btype='high')
        lead_filtered = filtfilt(b, a, lead)
    except:
        lead_filtered = lead
    
    # Step 2: Normalize signal for robust peak detection
    # Z-score normalization
    mean_signal = np.mean(lead_filtered)
    std_signal = np.std(lead_filtered)
    if std_signal > 0:
        lead_normalized = (lead_filtered - mean_signal) / std_signal
    else:
        lead_normalized = lead_filtered
    
    # Step 3: Detect R peaks with adaptive parameters
    # For 100 Hz sampling:
    # - Min heart rate: 40 bpm → 1.5s between beats → 150 samples
    # - Max heart rate: 180 bpm → 0.33s between beats → 33 samples
    # - Normal: 60-100 bpm → 0.6-1.0s → 60-100 samples
    
    min_distance = 35  # 0.35s minimum (170 bpm max)
    
    # Use normalized signal for peak detection
    # R-peaks should be positive in normalized signal
    peaks, properties = find_peaks(
        lead_normalized,
        height=0.5,  # At least 0.5 std above mean (normalized)
        distance=min_distance,
        prominence=0.4  # Require clear prominence
    )
    
    # Validation: Check if we found reasonable number of peaks
    # For 10-second ECG at 60-100 bpm, expect 10-17 beats
    if len(peaks) < 5:
        # Try with more lenient parameters
        peaks, _ = find_peaks(
            lead_normalized,
            height=0.3,
            distance=min_distance,
            prominence=0.2
        )
    
    if len(peaks) < 3:
        # Still not enough peaks - signal too noisy or flat
        return {
            'has_arrhythmia': False,
            'rr_coefficient_variation': 0.0,
            'mean_rr': 0.0,
            'confidence': 0.3,
            'num_peaks': len(peaks),
            'reason': f'Insufficient peaks detected ({len(peaks)})'
        }
    
    # Step 4: Calculate RR intervals (in samples)
    rr_intervals = np.diff(peaks)
    
    # Remove outliers (potential false peaks)
    # RR intervals should be between 33-200 samples (0.33-2.0 seconds at 100Hz)
    valid_rr = rr_intervals[(rr_intervals >= 33) & (rr_intervals <= 200)]
    
    if len(valid_rr) < 2:
        return {
            'has_arrhythmia': False,
            'rr_coefficient_variation': 0.0,
            'mean_rr': 0.0,
            'confidence': 0.3,
            'num_peaks': len(peaks),
            'reason': 'Invalid RR intervals detected'
        }
    
    # Step 5: Statistical analysis
    mean_rr = np.mean(valid_rr)
    std_rr = np.std(valid_rr)
    cv_rr = (std_rr / mean_rr) if mean_rr > 0 else 0  # Coefficient of Variation
    
    # Step 6: Clinical thresholds
    # CV < 0.08 (8%) = Normal sinus rhythm (stable RR)
    # CV > 0.12 (12%) = Likely arrhythmia (variable RR)
    # 0.08-0.12 = Borderline
    
    if cv_rr < 0.08:
        has_arrhythmia = False
        confidence = 0.85  # High confidence in normal
    elif cv_rr > 0.12:
        has_arrhythmia = True
        confidence = 0.80  # High confidence in arrhythmia
    else:
        # Borderline - defer to ML model
        has_arrhythmia = None  # Will use model prediction
        confidence = 0.50
    
    # Calculate heart rate for additional info
    mean_hr = (fs / mean_rr) * 60  # beats per minute
    
    return {
        'has_arrhythmia': has_arrhythmia,
        'rr_coefficient_variation': cv_rr,
        'mean_rr': mean_rr,
        'mean_hr': mean_hr,
        'confidence': confidence,
        'num_peaks': len(peaks),
        'num_valid_rr': len(valid_rr),
        'reason': f'RR CV: {cv_rr:.3f}'
    }

def preprocess_ecg(df):
    # Expecting (1000, 12) or (187, 1) or similar variations
    # Model Input: (B, 12, 1000)
    
    data = df.values # Numpy array
    
    # Scenario 1: User uploads proper PTB-XL chunk or similar (N, 12)
    # Check dimensions
    if data.shape[1] == 12:
         # Check length
         if data.shape[0] != 1000:
             # Resize to 1000 length
             # cv2.resize expects (width, height). We want to resize time axis (rows)
             # Input to cv2.resize: (12, N) -> (12, 1000)
             # Wait, cv2.resize works on images.
             # Let's use scipy or just linear interpolation. Or cv2
             # cv2 resize src=(W, H)
             resampled = []
             for i in range(12):
                 # Resize individual channel
                 channel_data = data[:, i]
                 # Resize to 1000
                 current_len = len(channel_data)
                 resampled_channel = np.interp(
                     np.linspace(0, current_len, 1000),
                     np.arange(current_len),
                     channel_data
                 )
                 resampled.append(resampled_channel)
             signal = np.array(resampled) # (12, 1000)
         else:
             signal = data.T # (12, 1000)
             
    # Scenario 2: Single Lead (e.g. MIT-BIH 187 length)
    elif data.shape[1] == 1 or len(data.shape) == 1:
        # Flatten
        flat_data = data.flatten()
        # Resize to 1000
        current_len = len(flat_data)
        resampled_channel = np.interp(
             np.linspace(0, current_len, 1000),
             np.arange(current_len),
             flat_data
         )
        # Duplicate to 12 channels (Simple fallback)
        signal = np.tile(resampled_channel, (12, 1)) # (12, 1000)
        
    else:
        st.error(f"Unexpected data shape: {data.shape}. Expected 12 columns (leads).")
        return None, None

    # Z-Score Normalization REMOVED to match training pipeline
    # The model was trained on raw WFDB values (mV)
    # mean = np.mean(signal, axis=1, keepdims=True)
    # std = np.std(signal, axis=1, keepdims=True)
    # signal = (signal - mean) / (std + 1e-6)
    
    # To Tensor (B, C, L)
    tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
    return tensor, signal # Signal is (12, 1000)


def create_pdf_report(prediction, confidence, model_name, rr_analysis=None, decision_basis=None):
    class PDF(FPDF):
        def header(self):
            # Banner
            self.set_fill_color(22, 27, 34) # Dark Background
            self.rect(0, 0, 210, 40, 'F')
            # Logo Text
            self.set_xy(10, 10)
            self.set_font('Arial', 'B', 24)
            self.set_text_color(0, 201, 255) # Cyan
            self.cell(0, 10, 'NeuroX', 0, 0, 'L')
            # Subtitle
            self.set_xy(10, 20)
            self.set_font('Arial', '', 11)
            self.set_text_color(200, 200, 200)
            self.cell(0, 10, 'AI-Powered Medical Diagnostics', 0, 0, 'L')

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, 'Generated by NeuroX AI | Research Use Only', 0, 0, 'C')

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # Push content below header
    pdf.set_y(50)
    
    # Metadata Section
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Patient Examination Details", ln=True)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    
    pdf.set_font("Arial", '', 11)
    # Use a tabular alignment using cell width
    pdf.cell(30, 8, "Date:", 0, 0)
    pdf.cell(0, 8, pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'), ln=True)
    pdf.cell(30, 8, "Model:", 0, 0)
    pdf.cell(0, 8, model_name, ln=True)
    pdf.cell(30, 8, "ID:", 0, 0)
    pdf.cell(0, 8, f"N-{np.random.randint(1000, 9999)}", ln=True)
    pdf.ln(10)
    
    # Diagnosis Box
    box_y = pdf.get_y()
    pdf.set_fill_color(245, 247, 250) # Very light gray/blue
    pdf.rect(10, box_y, 190, 45, 'F')
    
    pdf.set_xy(15, box_y + 5)
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "DIAGNOSTIC RESULT", ln=True)
    
    # Condition Color
    if prediction == "NORMAL":
        r, g, b = 46, 160, 67 # Green
    else:
        r, g, b = 255, 123, 114 # Red
        
    pdf.set_xy(15, pdf.get_y())
    pdf.set_font("Arial", 'B', 24)
    pdf.set_text_color(r, g, b)
    pdf.cell(0, 12, prediction, ln=True)
    
    pdf.set_xy(15, pdf.get_y())
    pdf.set_font("Arial", '', 12)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 8, f"Confidence Score: {confidence}", ln=True)
    
    pdf.set_y(box_y + 45 + 10) # Move below box
    
    # Clinical Analysis Section (for ECG reports)
    if rr_analysis:
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Clinical Analysis", ln=True)
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        # Clinical metrics table
        pdf.set_font("Arial", '', 11)
        
        # RR Variability
        pdf.cell(60, 8, "RR Variability:", 0, 0)
        rr_cv = rr_analysis.get('rr_coefficient_variation', 0) * 100
        pdf.cell(0, 8, f"{rr_cv:.1f}%", ln=True)
        
        # Heart Rate
        pdf.cell(60, 8, "Heart Rate:", 0, 0)
        hr = rr_analysis.get('mean_hr', 0)
        pdf.cell(0, 8, f"{hr:.0f} bpm", ln=True)
        
        # R Peaks
        pdf.cell(60, 8, "R Peaks Detected:", 0, 0)
        peaks = rr_analysis.get('num_peaks', 'N/A')
        pdf.cell(0, 8, f"{peaks}", ln=True)
        
        # Decision Basis
        if decision_basis:
            pdf.cell(60, 8, "Decision Basis:", 0, 0)
            pdf.cell(0, 8, decision_basis, ln=True)
        
        pdf.ln(5)
        
        # Clinical note
        pdf.set_font("Arial", 'I', 9)
        pdf.set_text_color(100, 100, 100)
        pdf.multi_cell(0, 5, f"Clinical Note: RR Coefficient of Variation < 8% indicates stable rhythm (Normal), > 12% indicates irregular rhythm (Arrhythmia). Current measurement: {rr_cv:.1f}%")
        
        pdf.ln(5)
    
    # Disclaimer
    pdf.set_text_color(100, 100, 100)
    pdf.set_font("Arial", 'I', 9)
    pdf.multi_cell(0, 5, "Disclaimer: This report was generated automatically by an AI system. It is intended to assist medical professionals and should not be used as the sole basis for clinical diagnosis. Please review these results with a qualified specialist.")
    
    return pdf.output(dest='S').encode('latin-1')

# --- Pages ---

if page == "Home":
    st.markdown("""
    <div style='margin-bottom: 20px;'>
        <h1 style='margin-bottom: 5px; background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Next-Gen Medical Diagnostics</h1>
        <h3 style='color: #8B949E; font-weight: 400; margin-top: 0px;'>AI-Powered Analysis for X-Ray & ECG</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background-color: #161B22; padding: 25px; border-radius: 15px; border: 1px solid #30363D; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h3 style='color: #00C9FF;'>Pneumonia Detection</h3>
            <p style='color: #8B949E; margin-bottom: 20px;'>
                Analyze Chest X-Ray images using <strong>EfficientNet-B0</strong> (Enhanced with CLAHE). 
                Includes Gradient-weighted Class Activation Mapping (Grad-CAM) for localization.
            </p>
            <ul style='color: #C9D1D9;'>
                <li><strong>95.8%</strong> Accuracy</li>
                <li>< 200ms Inference Time</li>
                <li>Heatmap Visualization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div style='background-color: #161B22; padding: 25px; border-radius: 15px; border: 1px solid #30363D; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h3 style='color: #FF7B72;'>Arrhythmia Detection</h3>
            <p style='color: #8B949E; margin-bottom: 20px;'>
                 Classify cardiac rhythms from 12-lead or single-lead ECG signals using <strong>ResNet1D-18</strong>.
                 Robust categorization of Normal vs. Arrhythmia.
            </p>
            <ul style='color: #C9D1D9;'>
                <li>Trained on PTB-XL Database</li>
                <li>Noise-Resistant Architecture</li>
                <li>Instant Risk Assessment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 30px'></div>", unsafe_allow_html=True)
    
    with st.expander("ℹ️ How to Use This System", expanded=False):
        st.markdown("""
        1. **Select a Module** from the top navigation bar (Pneumonia or Arrhythmia).
        2. **Upload Medical Data**:
            - For X-Ray: Upload a `.jpg` or `.png` chest radiograph.
            - For ECG: Upload a `.csv` file containing lead signal data.
        3. **View Analysis**: The AI will generate a diagnosis, confidence score, and explainability map.
        4. **Download Report**: Save the results as a medical-grade PDF.
        """)
        
    st.markdown("---")
    st.warning("**DISCLAIMER**: This system is for **research and educational purposes only**. It is not intended for clinical diagnosis, treatment, or medical decision-making.")

elif page == "Pneumonia Detection":
    st.title("Chest X-Ray Analysis")
    
    # Pre-load Model
    load_xray_model()
    
    # --- Logic to Swap Views ---
    # --- Logic to Swap Views ---
    if not st.session_state.get('analyze_xray'):
        with st.container():
            # --- VIEW 1: Input ---
            st.markdown("### Input Image")
            uploaded_file = st.file_uploader("Upload X-Ray (PA/AP View)", type=["jpg", "jpeg", "png"], key="xray_uploader")
            
            if uploaded_file:
                # Automatically Persist File Persistence (BytesIO)
                # Fixes potential closed file error when widget is removed
                if 'persistent_prod_file_bytes' not in st.session_state or \
                   st.session_state.get('last_uploaded_name') != uploaded_file.name:
                    
                    # Read into new BytesIO
                    import io
                    bytes_data = uploaded_file.getvalue()
                    st.session_state['persistent_prod_file_bytes'] = io.BytesIO(bytes_data)
                    st.session_state['last_uploaded_name'] = uploaded_file.name

                def start_analysis():
                    st.session_state['analyze_xray'] = True
                    
                st.button("Analyze X-Ray", width="stretch", key="xray_btn", on_click=start_analysis)
            
            if not uploaded_file:
                 st.info("Please upload a chest X-ray image to start.")
    else:
        with st.container():
            # --- VIEW 2: Results Analysis ---
            # Retrieve the persistent bytes
            file_buffer = st.session_state.get("persistent_prod_file_bytes")
            
            if file_buffer:
                col_header, col_back = st.columns([3, 1])
                with col_header:
                    st.markdown("### Analysis Result")
                with col_back:
                    def reset_analysis():
                        st.session_state['analyze_xray'] = False
                        # Optional: Clear persistent file if desired
                        # st.session_state.pop('persistent_prod_file_bytes', None)
                    st.button("New Scan", on_click=reset_analysis, key="back_btn")
                
                # Original Image Loading from Buffer
                file_buffer.seek(0)
                original_image = Image.open(file_buffer).convert('RGB')
                display_image = original_image.copy()

                # Smart Contrast Toggle
                use_clahe = st.toggle("Enable Smart Contrast (CLAHE)", value=False, key="clahe_toggle_results")
                
                if use_clahe:
                    img_np = np.array(original_image)
                    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
                    enhanced = clahe.apply(gray)
                    display_image = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))

                # --- RESULT COMPUTATION (RUN ONCE) ---
                # We check if we already have results for this specific file upload
                # We use a composite key of filename + 'xray_result' to ensure validity
                result_key = f"xray_result_{st.session_state.get('last_uploaded_name')}"
                
                if result_key not in st.session_state:
                    with st.spinner("Analyzing Radiograph..."):
                        model, device = load_xray_model()
                        if model:
                            try:
                                # Inference
                                img_tensor = preprocess_image(original_image).to(device)
                                outputs = model(img_tensor)
                                prob = torch.sigmoid(outputs).item()
                                prediction = "PNEUMONIA" if prob > 0.5 else "NORMAL"
                                confidence = prob if prob > 0.5 else 1 - prob
                                
                                # Explainability
                                target_layer = model.features[8]
                                grad_cam = GradCAM(model, target_layer)
                                cam, _ = grad_cam(img_tensor)
                                heatmap_img_np = overlay_cam(img_tensor[0], cam)
                                
                                # Process Heatmap for Display
                                heatmap_pil = Image.fromarray((heatmap_img_np * 255).astype(np.uint8))
                                heatmap_pil = heatmap_pil.resize(original_image.size, Image.Resampling.BICUBIC)
                                
                                # Store in Session State
                                st.session_state[result_key] = {
                                    "prediction": prediction,
                                    "confidence": confidence,
                                    "heatmap": heatmap_pil,
                                    "timestamp": pd.Timestamp.now()
                                }
                                
                                # Persist to Database
                                add_record(
                                    modality="X-Ray",
                                    filename=st.session_state.get('last_uploaded_name', 'unknown.png'),
                                    result=prediction,
                                    confidence=float(confidence),
                                    details="MobileNetV2 Analysis"
                                )
                                
                            except Exception as e:
                                st.error(f"Analysis Failed: {e}")
                                st.stop()

                # --- RESULT RENDERING (FROM CACHE) ---
                if result_key in st.session_state:
                    res = st.session_state[result_key]
                    prediction = res['prediction']
                    confidence = res['confidence']
                    heatmap_pil = res['heatmap']
                    
                    # --- Layout: Side-by-Side Analysis ---
                    col_img, col_metrics = st.columns([1, 1])
                    
                    with col_img:
                        st.markdown("### X-Ray Scan")
                        st.image(display_image, caption=f"View: {'Enhanced (CLAHE)' if use_clahe else 'Standard'}", use_container_width=True)
                    
                    with col_metrics:
                        st.markdown("### AI Attention")
                        st.image(heatmap_pil, caption="Grad-CAM Heatmap", use_container_width=True)

                    st.markdown("---")
                    
                    # Metrics Section
                    st.markdown("### Diagnostic Metrics")
                    color = "#FF7B72" if prediction == "PNEUMONIA" else "#2EA043"
                
                    m1, m2 = st.columns([1, 1])
                    with m1:
                        st.markdown(f"""
                        <div style="background-color: #161B22; padding: 15px; border-radius: 10px; border-left: 5px solid {color};">
                            <small style="color: #8B949E; font-weight: bold;">PREDICTION</small>
                            <h2 style="margin:0; color: {color}; font-size: 28px;">{prediction}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with m2:
                        st.markdown(f"""
                        <div style="background-color: #161B22; padding: 15px; border-radius: 10px; border: 1px solid #30363D;">
                            <small style="color: #8B949E; font-weight: bold;">CONFIDENCE SCORE</small>
                            <div style="display: flex; align-items: center; justify-content: space-between;">
                                <span style="font-size: 24px; font-weight: bold; color: #C9D1D9;">{confidence*100:.1f}%</span>
                                <div style="width: 60%; background-color: #21262D; height: 6px; border-radius: 3px;">
                                    <div style="width: {confidence*100}%; background-color: #00C9FF; height: 6px; border-radius: 3px;"></div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Report Section
                    st.markdown("<div style='height: 25px'></div>", unsafe_allow_html=True) 
                    pdf_bytes = create_pdf_report(prediction, f"{confidence*100:.2f}%", "EfficientNet-B0")
                    st.download_button(
                        label="📄 Download Diagnostic Report",
                        data=pdf_bytes,
                        file_name="neurox_report.pdf",
                        mime="application/pdf",
                        width="stretch",
                        key="xray_pdf_btn"
                    )
                    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True) 

                    # Update History (Once)
                    if st.session_state.get("last_history_timestamp") != res['timestamp']:
                         st.session_state['history'].append({
                            "type": "Chest X-Ray",
                            "result": prediction,
                            "conf": f"{confidence*100:.1f}%",
                            "color": color,
                            "timestamp": res['timestamp']
                        })
                         st.session_state["last_history_timestamp"] = res['timestamp']

elif page == "Arrhythmia Detection":
    st.title("ECG Arrhythmia Analysis")
    
    # Pre-load Model
    load_ecg_model()
    
    # --- VIEW LOGIC ---
    if not st.session_state.get('analyze_ecg'):
        with st.container():
            # --- Input View ---
            st.markdown("### Input Signal")
            uploaded_file = st.file_uploader("Upload ECG (CSV)", type=["csv"], key="ecg_file")
            
            if uploaded_file:
                 # Persist file bytes
                 if 'persistent_ecg_bytes' not in st.session_state or \
                    st.session_state.get('last_uploaded_ecg_name') != uploaded_file.name:
                     bytes_data = uploaded_file.getvalue()
                     st.session_state['persistent_ecg_bytes'] = io.BytesIO(bytes_data)
                     st.session_state['last_uploaded_ecg_name'] = uploaded_file.name

                 def start_ecg_analysis():
                    st.session_state['analyze_ecg'] = True
                    
                 st.button("Analyze Signal", width="stretch", key="ecg_btn", on_click=start_ecg_analysis)
                    
            if not uploaded_file and 'persistent_ecg_bytes' not in st.session_state:
                 st.info("Please upload a 1-lead ECG CSV file.")
    else:
        with st.container():
            # --- Results View ---
            col_header, col_back = st.columns([3, 1])
            with col_header:
                st.markdown("### Analysis Result")
            with col_back:
                def reset_ecg_analysis():
                    st.session_state['analyze_ecg'] = False
                st.button("New Scan", on_click=reset_ecg_analysis, key="ecg_back_btn")
            
            # Retrieve persistent bytes
            file_buffer = st.session_state.get('persistent_ecg_bytes')
            
            if file_buffer:
                try:
                    file_buffer.seek(0) # IMPORTANT: Reset buffer pointer
                    df = pd.read_csv(file_buffer, header=None)
                    
                    # --- RESULT COMPUTATION (RUN ONCE) ---
                    ecg_result_key = f"ecg_result_{st.session_state.get('last_uploaded_ecg_name')}"
                    
                    if ecg_result_key not in st.session_state:
                        with st.spinner("Processing ECG Signal..."):
                            model, device = load_ecg_model()
                            if model:
                                try:
                                    tensor, signal_array = preprocess_ecg(df)
                                    if tensor is None:
                                        st.stop()
                                    
                                    tensor = tensor.to(device)
                                    
                                    # CLINICAL ANALYSIS: RR Interval Check (Primary)
                                    rr_analysis = analyze_rr_intervals(signal_array)
                                    
                                    # Model Inference (Secondary)
                                    output = model(tensor)
                                    prob = torch.sigmoid(output).item()
                                    model_prediction = "ARRHYTHMIA" if prob > 0.65 else "NORMAL"  # Raised threshold to 65%
                                    model_confidence = prob if prob > 0.65 else 1 - prob
                                    
                                    # HYBRID DECISION: Clinical rules override weak model predictions
                                    if rr_analysis['has_arrhythmia'] is not None:
                                        # RR analysis is conclusive - use it
                                        prediction = "ARRHYTHMIA" if rr_analysis['has_arrhythmia'] else "NORMAL"
                                        confidence = rr_analysis['confidence']
                                        decision_basis = f"RR Analysis ({rr_analysis['reason']})"
                                    else:
                                        # RR borderline - use model but reduce confidence
                                        prediction = model_prediction
                                        confidence = model_confidence * 0.8  # Reduce confidence for borderline cases
                                        decision_basis = f"Model (RR borderline: {rr_analysis['reason']})"
                                    
                                    # Explainability
                                    target_layer = model.layer4[-1] 
                                    grad_cam = GradCAM(model, target_layer)
                                    cam, _ = grad_cam(tensor)
                                    
                                    # Store in Session State
                                    st.session_state[ecg_result_key] = {
                                        "prediction": prediction,
                                        "confidence": confidence,
                                        "signal_array": signal_array,
                                        "cam": cam,
                                        "tensor": tensor.cpu(), # Store on CPU to save GPU RAM
                                        "timestamp": pd.Timestamp.now(),
                                        "rr_analysis": rr_analysis,
                                        "decision_basis": decision_basis
                                    }
                                    
                                    # Persist to Database
                                    add_record(
                                        modality="ECG",
                                        filename=st.session_state.get('last_uploaded_ecg_name', 'unknown_ecg.csv'),
                                        result=prediction,
                                        confidence=float(confidence),
                                        details="ResNet1D-18 Analysis"
                                    )
                                except Exception as e:
                                    st.error(f"ECG Analysis Failed: {e}")
                                    st.stop()

                    # --- RESULT RENDERING (FROM CACHE) ---
                    if ecg_result_key in st.session_state:
                        res = st.session_state[ecg_result_key]
                        prediction = res['prediction']
                        confidence = res['confidence']
                        signal_array = res['signal_array']
                        cam = res['cam']
                        
                        color = "#2EA043" if prediction == "NORMAL" else "#FF7B72"

                        # --- Results Layout ---
                        st.markdown("### Diagnosis")
                        
                        # Metrics Column
                        m_col1, m_col2 = st.columns(2)
                        with m_col1:
                             st.markdown(f"""
                            <div class="metric-card" style="border-left: 4px solid {color};">
                                <div class="metric-label">Rhythm</div>
                                <div class="metric-value" style="color: {color};">{prediction}</div>
                            </div>""", unsafe_allow_html=True)
                        with m_col2:
                             st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Confidence</div>
                                <div class="metric-value">{confidence*100:.1f}%</div>
                            </div>""", unsafe_allow_html=True)
                        
                        # Clinical Details
                        rr_info = res.get('rr_analysis', {})
                        decision_basis = res.get('decision_basis', 'Model prediction')
                        
                        st.markdown("### Clinical Analysis")
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">RR Variability</div>
                                <div class="metric-value" style="font-size: 20px;">{rr_info.get('rr_coefficient_variation', 0)*100:.1f}%</div>
                            </div>""", unsafe_allow_html=True)
                        with c2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Heart Rate</div>
                                <div class="metric-value" style="font-size: 20px;">{rr_info.get('mean_hr', 0):.0f} bpm</div>
                            </div>""", unsafe_allow_html=True)
                        with c3:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">R Peaks</div>
                                <div class="metric-value" style="font-size: 20px;">{rr_info.get('num_peaks', 'N/A')}</div>
                            </div>""", unsafe_allow_html=True)
                        with c4:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Decision Basis</div>
                                <div class="metric-value" style="font-size: 12px; color: #8B949E;">{decision_basis}</div>
                            </div>""", unsafe_allow_html=True)
                        
                        st.info(f"ℹ️ **Clinical Note:** RR CV < 8% indicates stable rhythm (Normal), > 12% indicates irregular rhythm (Arrhythmia). Current: {rr_info.get('rr_coefficient_variation', 0)*100:.1f}%")
                            
                        # Signal & CAM
                        st.markdown("### Rhythm Visualization (Lead II)")
                        
                        lead_ii = signal_array[1] if signal_array.shape[0] > 1 else signal_array[0]
                        
                        fig, ax = plt.subplots(figsize=(10, 3))
                        fig.patch.set_facecolor('#161B22')
                        ax.set_facecolor('#0E1117')
                        ax.plot(lead_ii, color='#00C9FF', linewidth=1.5, label='Lead II')
                        
                        # CAM Visualization
                        extent = [0, len(lead_ii), np.min(lead_ii), np.max(lead_ii)]
                        cam_expanded = np.expand_dims(cam, axis=0)
                        
                        ax.imshow(cam_expanded, aspect='auto', cmap='Reds', alpha=0.6, extent=extent, interpolation='bilinear')
                        
                        for spine in ax.spines.values():
                            spine.set_color('#30363D')
                        ax.tick_params(colors='#8B949E')
                        
                        # Render to Buffer
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", bbox_inches='tight', facecolor=fig.get_facecolor())
                        buf.seek(0)
                        st.image(buf, width="stretch") # Using 'stretch' per recommendation
                        plt.close(fig)

                        # Report Section
                        st.markdown("<div style='height: 25px'></div>", unsafe_allow_html=True)
                        pdf_bytes = create_pdf_report(
                            prediction, 
                            f"{confidence*100:.2f}%", 
                            "ECGNet (ResNet1D-18)",
                            rr_analysis=rr_info,
                            decision_basis=decision_basis
                        )
                        st.download_button(
                            label="📄 Download Diagnostic Report",
                            data=pdf_bytes,
                            file_name="neurox_ecg.pdf",
                            mime="application/pdf",
                            width="stretch",
                            key="ecg_pdf_btn"
                        )

                        # History Logic (Once)
                        if st.session_state.get("last_history_ecg_timestamp") != res['timestamp']:
                             st.session_state['history'].append({
                                "type": "ECG Signal",
                                "result": prediction,
                                "conf": f"{confidence*100:.1f}%",
                                "color": color,
                                "timestamp": res['timestamp']
                            })
                             st.session_state["last_history_ecg_timestamp"] = res['timestamp']

                except Exception as e:
                     st.error(f"Error processing signal: {e}")




elif page == "Dashboard":
    st.markdown("""
    <div style='text-align: center; margin-bottom: 30px; margin-top: -30px;'>
        <h2 style='color: #00C9FF;'>Patient History Dashboard</h2>
        <p style='color: #8B949E;'>Overview of diagnostic activity and past records</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    total_scans, anomalies = get_stats()
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Scans</div>
            <div class="metric-value">{total_scans}</div>
        </div>""", unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #FF7B72;">
            <div class="metric-label">Anomalies Detected</div>
            <div class="metric-value">{anomalies}</div>
        </div>""", unsafe_allow_html=True)
        
    with c3:
        percentage = (anomalies/total_scans*100) if total_scans > 0 else 0
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #D2A8FF;">
            <div class="metric-label">Disease Prevalence</div>
            <div class="metric-value">{percentage:.1f}%</div>
        </div>""", unsafe_allow_html=True)
        
    st.markdown("### 📜 Recent Activity")
    df = get_history()
    
    if not df.empty:
        # Styling the dataframe
        st.dataframe(
            df[['timestamp', 'modality', 'result', 'confidence', 'filename']],
            column_config={
                "timestamp": "Date & Time",
                "modality": "Scan Type",
                "result": "Diagnosis",
                "confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
                "filename": "File Name"
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No records found. Run an analysis to populate the dashboard.")

elif page == "About":
    st.title("Project Specifications")
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("### 🧠 Neural Network Architectures")
        st.markdown("""
        <div style='background-color: #161B22; padding: 20px; border-radius: 10px; border-left: 4px solid #00C9FF; margin-bottom: 20px;'>
            <strong style='color: #E6EDF3; font-size: 18px;'>Vision Model (X-Ray)</strong><br>
            <span style='color: #8B949E;'>EfficientNet-B0 + Smart Contrast</span>
            <ul style='color: #C9D1D9; margin-top: 10px;'>
                <li>Pre-trained on ImageNet, Fine-tuned on Chest X-Rays</li>
                <li><strong>95.8%</strong> Test Accuracy</li>
                <li>Contrast Limited Adaptive Histogram Equalization (CLAHE)</li>
            </ul>
        </div>
        
        <div style='background-color: #161B22; padding: 20px; border-radius: 10px; border-left: 4px solid #FF7B72; margin-bottom: 20px;'>
            <strong style='color: #E6EDF3; font-size: 18px;'>Time-Series Model (ECG)</strong><br>
            <span style='color: #8B949E;'>ResNet1D-18</span>
            <ul style='color: #C9D1D9; margin-top: 10px;'>
                <li>Deep Residual Network adapted for 1D signals</li>
                <li>Trained on PTB-XL (PhysioNet) Database</li>
                <li>Robust against noisy signal artifacts</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### 🛠 Tech Stack")
        st.markdown("""
        <div class="metric-card">
            <strong style='color: #E6EDF3;'>Core Frameworks</strong>
            <p style='color: #8B949E; margin: 5px 0 0 0;'>PyTorch 2.0 • CUDA 11.8</p>
        </div>
        <div class="metric-card">
            <strong style='color: #E6EDF3;'>Interface</strong>
            <p style='color: #8B949E; margin: 5px 0 0 0;'>Streamlit • Matplotlib • FPDF</p>
        </div>
        <div class="metric-card">
            <strong style='color: #E6EDF3;'>Explainability</strong>
            <p style='color: #8B949E; margin: 5px 0 0 0;'>Grad-CAM • Component Activation</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #8B949E; padding: 20px;'>
        <p>Built for Advanced AI Research Project</p>
        <p>© 2026 NeuroX Diagnostics</p>
    </div>
    """, unsafe_allow_html=True)
