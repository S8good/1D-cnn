import pandas as pd
import re
import os

csv_path = r"c:\Users\Spc\Desktop\3.LSPR-code\LSPR_code\DeepLearning\LSPR_Spectra_Master\data\processed\Reconstructed_Preprocessed_Spectra.csv"

if not os.path.exists(csv_path):
    print("File not found")
else:
    df = pd.read_csv(csv_path)
    wavelengths = df['Wavelength'].values
    print(f"seq_len: {len(wavelengths)}")
    
    count = 0
    for col in df.columns:
        if col == 'Wavelength': continue
        if 'Ag' not in col: continue
        match = re.search(r'([0-9.]+)\s*ng/ml', col)
        if match:
            conc = float(match.group(1))
            if count < 5:
                print(f"Col: {col} -> extracted conc: {conc}")
            count += 1
            
    print(f"Total Ag spectra with concentration: {count}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"PyTorch import failed: {e}")
