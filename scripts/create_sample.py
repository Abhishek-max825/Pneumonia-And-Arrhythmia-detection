import wfdb
import pandas as pd
import numpy as np
import os
import ast

def create_samples(samples_per_class=100, output_parent="ecg_testing"):
    # Check multiple possible locations
    possible_paths = [r"ecg/ptbxl_database.csv", r"data/ecg/ptbxl_database.csv"]
    db_path = None
    for p in possible_paths:
        if os.path.exists(p):
            db_path = p
            break
            
    if not db_path:
        print(f"❌ Database not found in {possible_paths}")
        return

    df = pd.read_csv(db_path)
    base_dir = os.path.dirname(db_path)
    
    # Create subdirectories
    normal_dir = os.path.join(output_parent, "Normal_Sinus_Rhythm")
    arrhythmia_dir = os.path.join(output_parent, "Arrhythmia_Rhythm")
    
    for d in [normal_dir, arrhythmia_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")

    count_norm = 0
    count_abnorm = 0
    
    print(f"Generating samples int {output_parent}...")
    
    for idx, row in df.iterrows():
        # Stop if we have enough of both
        if count_norm >= samples_per_class and count_abnorm >= samples_per_class:
            break
            
        try:
            # Parse SCP codes
            # Example: "{'NORM': 100.0, 'SR': 0.0}"
            scp_dict = ast.literal_eval(row['scp_codes'])
            
            is_normal = 'NORM' in scp_dict
            
            # Determine target subdirectory
            if is_normal:
                if count_norm >= samples_per_class: continue
                target_dir = normal_dir
            else:
                if count_abnorm >= samples_per_class: continue
                target_dir = arrhythmia_dir
            
            filename = row['filename_lr']
            full_path_base = os.path.join(base_dir, filename)
            
            if os.path.exists(full_path_base + ".dat"):
                signal, metadata = wfdb.rdsamp(full_path_base)
                # Output filename
                fname = os.path.basename(filename) + ".csv"
                out_path = os.path.join(target_dir, fname)
                
                # Save without header/index
                pd.DataFrame(signal).to_csv(out_path, header=False, index=False)
                
                if is_normal:
                    count_norm += 1
                else:
                    count_abnorm += 1
                    
                if (count_norm + count_abnorm) % 10 == 0:
                    print(f"  ...generated {count_norm} Normal, {count_abnorm} Arrhythmia")
                    
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
                
    print(f"✅ Completed! Saved to '{output_parent}/'")
    print(f"   - Normal: {count_norm}")
    print(f"   - Arrhythmia: {count_abnorm}")

if __name__ == "__main__":
    create_samples()
