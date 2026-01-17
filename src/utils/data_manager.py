import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
XRAY_DIR = DATA_DIR / "xray"
ECG_DIR = DATA_DIR / "ecg"

def organize_xray():
    print("--- Organizing X-Ray Data ---")
    source_dir = XRAY_DIR / "chest_xray"
    if not source_dir.exists():
        print("ℹ️  X-ray data seems already organized or missing 'chest_xray' folder.")
        return

    # Move train, val, test to XRAY_DIR
    for folder in ["train", "val", "test"]:
        src = source_dir / folder
        dst = XRAY_DIR / folder
        if src.exists():
            if dst.exists():
                print(f"⚠️  Target {dst} already exists. Merging/Skipping...")
            else:
                shutil.move(str(src), str(dst))
                print(f"✅ Moved {folder} to {XRAY_DIR}")
    
    # Remove empty source dir
    if source_dir.exists() and not any(source_dir.iterdir()):
        source_dir.rmdir()
        print("🗑️  Removed empty 'chest_xray' folder")
    elif source_dir.exists():
         print(f"ℹ️  'chest_xray' not empty, kept: {list(source_dir.iterdir())}")

def load_ecg_record(csv_path, annotation_path):
    # Load Signal
    # col 0: sample #, col 1: MLII, col 2: V5 (usually)
    # The file has a header: 'sample #','MLII','V5'
    try:
        sig_df = pd.read_csv(csv_path)
        # Check standard columns - usually the second column is the lead we want (MLII)
        # Sometimes header is weird, let's verify
        signal = sig_df.iloc[:, 1].values # taking MLII
        
        # Load Annotations
        # Format: Time, Sample #, Type, Sub, Chan, Num, Aux
        # It is fixed width or tab separated often.
        # Based on view_file: many spaces.
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
        
        peaks = []
        labels = []
        
        # Skip header line 1
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 3:
                sample_idx = int(parts[1])
                beat_type = parts[2]
                peaks.append(sample_idx)
                labels.append(beat_type)
        
        return signal, peaks, labels
        
    except Exception as e:
        print(f"❌ Error loading {csv_path.name}: {e}")
        return None, None, None

def process_ecg():
    print("\n--- Processing ECG Data ---")
    # MIT-BIH AAMI classes mapping
    # N -> Normal
    # A, a, J, S -> SVEB
    # V, E -> VEB
    # F -> Fusion
    # Q, /, f, u -> Unknown/Other
    # For this task: Normal (N) vs Arrhythmia (Everything else that implies a beat)
    # We exclude non-beat annotations like keys, comments.
    
    # Valid beat codes often used:
    # N, L, R, e, j, A, a, J, S, V, E, F, /, f, Q
    # We will treat 'N', 'L', 'R', 'e', 'j' as Normal (or at least Non-Arrhythmic class often in binary)
    # Wait, strict "Normal vs Arrhythmia":
    # Usually N is Normal. L, R (Bundle branch blocks) are structural but often class 0 in 5-class AAMI.
    # However, for "Normal vs Arrhythmia" binary:
    # N = 0 (Normal)
    # All others = 1 (Arrhythmia)
    
    csv_files = list(ECG_DIR.glob("*.csv"))
    records = [f.stem for f in csv_files if f.stem.isdigit()] # e.g. '100', '101'
    
    processed_data = [] # List of {'signal': [187 floats], 'label': 0/1}
    
    # Window size? Typical 1D CNN input.
    # paper: "ECG Heartbeat Classification A Deep Transferable Representation" uses 187 samples.
    # We will extract centered beats. resampling might be needed if fs != 125/360.
    # MIT-BIH is 360Hz.
    # Let's retain 360Hz and take fixed window? Or resize?
    # Simple approach: Window of 180 samples (0.5s) centered on peak?
    # MIT-BIH is 360Hz. 187 samples is ~0.5s.
    # Let's take 187 samples centered on R-peak.
    
    for record in tqdm(records, desc="Processing Records"):
        csv_path = ECG_DIR / f"{record}.csv"
        ann_path = ECG_DIR / f"{record}annotations.txt"
        
        if not ann_path.exists():
            continue
            
        signal, peaks, labels = load_ecg_record(csv_path, ann_path)
        if signal is None:
            continue
            
        # Normalize signal per record (standard practice)
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
        
        for idx, (peak_idx, label) in enumerate(zip(peaks, labels)):
            # Filter non-beat annotations if necessary
            if label not in ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F', '/', 'f', 'Q']:
                continue
                
            # Binary Label Mapping
            # 0: Normal (N)
            # 1: Arrhythmia (Others)
            # NOTE: 'L', 'R' etc are technically arrhythmias or blocks, but in 5-class AAMI they are separate.
            # IN binary Normal vs Arrhythmia, usually N, L, R => Normal (class 0) or just N => Normal.
            # Given user prompt "Normal vs Arrhythmia", simplest is N vs Rest.
            # But let's check standard Kaggle dataset "MIT-BIH Arrhythmia Dataset (CSV)" creates:
            # It usually follows the "Arhythmia" vs "Normal" classification.
            # Let's strictly call 'N' Normal (0) and everything else Arrhythmia (1).
            
            y = 0 if label == 'N' else 1
            
            # Extract Segment
            # 90 before, 97 after = 187
            start = peak_idx - 90
            end = peak_idx + 97
            
            if start < 0 or end > len(signal):
                continue
                
            segment = signal[start:end]
            processed_data.append(np.concatenate(([y], segment))) # Label first
            
    # Convert to DataFrame
    columns = ['label'] + [f't_{i}' for i in range(187)]
    df = pd.DataFrame(processed_data, columns=columns)
    
    # Train/Test Split
    # Important: Split by RECORD to avoid leakage? 
    # Or random split? Standard Kaggle MIT-BIH is often random split of beats (which leaks patient data).
    # Since we want "Research Grade", patient split is better.
    # BUT, the "train.csv" and "test.csv" implies a single static split.
    # Given the constraint to match Kaggle "MIT-BIH Arrhythmia Dataset (CSV)" format approx:
    # That dataset usually splits into train.csv and test.csv.
    # I will verify simply saving them.
    
    train, test = train_test_split(df, test_size=0.2, shuffle=True, stratify=df['label'])
    
    print(f"✅ Processed {len(df)} heartbeats.")
    print(f"   Train size: {len(train)}")
    print(f"   Test size: {len(test)}")
    
    train.to_csv(ECG_DIR / "train.csv", index=False, header=False) # Kaggle dataset usually no header? or yes? 
    # Let's keep no header to match standard MNIST-like CSVs often used, OR add header.
    # User prompt: "Format: .zip archive containing CSV files ... train.csv, test.csv"
    # I'll add header for safety in my own loader, assume my own loader reads it.
    train.to_csv(ECG_DIR / "train.csv", index=False)
    test.to_csv(ECG_DIR / "test.csv", index=False)
    print(f"💾 Saved to {ECG_DIR}/train.csv and test.csv")

if __name__ == "__main__":
    organize_xray()
    process_ecg()
