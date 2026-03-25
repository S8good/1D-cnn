import sys
import os
import numpy as np

# 添加 src 到路径以便导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.ai_engine import get_ai_engine

if __name__ == "__main__":
    engine = get_ai_engine()
    
    conc_1 = 0.5
    conc_2 = 50.0
    conc_3 = 100.0
    
    spec_1 = engine.generate_spectrum(conc_1)
    spec_2 = engine.generate_spectrum(conc_2)
    spec_3 = engine.generate_spectrum(conc_3)
    
    diff_1_2 = np.max(np.abs(spec_1 - spec_2))
    diff_1_3 = np.max(np.abs(spec_1 - spec_3))
    
    print(f"Max difference between 0.5 and 50.0: {diff_1_2}")
    print(f"Max difference between 0.5 and 100.0: {diff_1_3}")
    
    # 打印前 5 个值对比
    print(f"Spec(0.5)  head: {spec_1[:5]}")
    print(f"Spec(50.0) head: {spec_2[:5]}")
