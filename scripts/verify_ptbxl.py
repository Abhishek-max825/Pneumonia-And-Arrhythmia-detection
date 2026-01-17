import pandas as pd
import wfdb
import ast
import numpy as np
import os

# --- PATHS ---
DATA_DIR = r"c:\Users\abhis\OneDrive\ドキュメント\Pneumonia_and_Heart_Disease_Detection\ecg"
DB_PATH = os.path.join(DATA_DIR, "ptbxl_database.csv")

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def verify_ecg_loading():
    print(f"📂 Checking Database at: {DB_PATH}")
    
    # 1. Load CSV
    try:
        df = pd.read_csv(DB_PATH, index_col='ecg_id')
        print(f"✅ Loaded Database. Shape: {df.shape}")
        
        # 2. Parse labels (scp_codes)
        df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))
        print("✅ Parsed scp_codes column.")
        
        # 3. Check first entry
        first_entry = df.iloc[0]
        filename = first_entry.filename_lr
        full_path = os.path.join(DATA_DIR, filename)
        
        print(f"🔍 Testing Record: {filename}")
        print(f"   Full Path: {full_path}")
        
        # 4. Read Signal with WFDB
        # Note: wfdb reads without extension for the header/dat pair
        # filename_lr is like 'records100/00000/00001_lr'
        # we need to pass absolute path without extension? Or relative to some path?
        # wfdb.rdsamp takes 'record_name'. If providing full path, it should work.
        
        signal, meta = wfdb.rdsamp(full_path)
        print(f"✅ Signal Loaded Successfully!")
        print(f"   Shape: {signal.shape} (Samples, Channels)")
        print(f"   Sampling Rate: {meta['fs']} Hz")
        print(f"   Signal Units: {meta['units']}")
        print(f"   Signal Channels: {meta['sig_name']}")
        
        # 5. Diagnostic Aggregation (Superclasses)
        # Standard PTB-XL mapping
        agg_df = pd.read_csv('https://raw.githubusercontent.com/physionet-org/ptb-xl/master/scp_statements.csv', index_col=0)
        # If we can't access internet, we might need a local mapping or hardcode it.
        # Let's see if we can just inspect the scp_codes first.
        print(f"   Labels (SCP): {first_entry.scp_codes}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        # Print detailed traceback if needed
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    verify_ecg_loading()
