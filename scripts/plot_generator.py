import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 添加 src 到路径以便导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.ai_engine import get_ai_engine

if __name__ == "__main__":
    engine = get_ai_engine()
    
    concs = [0.1, 1.0, 10.0, 50.0, 100.0]
    wavelengths = engine.get_wavelengths()
    
    plt.figure(figsize=(10, 6))
    for c in concs:
        spec = engine.generate_spectrum(c)
        plt.plot(wavelengths, spec, label=f'{c} ng/ml')
        
    plt.title("Generator Outputs for Different Concentrations")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Absorbance")
    plt.xlim(550, 650)
    plt.legend()
    plt.grid(True)
    
    out_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "generator_output.png")
    plt.savefig(out_path)
    print(f"Saved to {out_path}")
