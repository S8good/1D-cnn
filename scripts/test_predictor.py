import sys
import os
import pandas as pd
import numpy as np

# 添加 src 到路径以便导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.ai_engine import get_ai_engine

if __name__ == "__main__":
    engine = get_ai_engine()
    
    # 获取真实数据
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "filtered", "Hill_Filtered_Paired_Spectra_20pct.xlsx")
    df = pd.read_excel(data_path)
    
    print("Testing Predictor with exact training data strips...")
    
    # 抽取 0.5, 1.0, 50.0 浓度的数据进行预测
    test_cols = [c for c in df.columns if 'Ag-01' in c or 'Ag-02' in c]
    
    for col in test_cols[:10]: # 测试前 10 个条目
        # 提取真实浓度 (从列名比如 0.5ng/ml-Ag-01_xx)
        import re
        match = re.search(r'([0-9.]+)\s*ng/ml', col)
        if not match: continue
        true_conc = float(match.group(1))
        
        # 提取光谱
        spectra = df[col].values
        
        # 预测
        pred_conc = engine.predict_concentration(spectra)
        
        print(f"Col: {col[:15]}... | True: {true_conc:5.2f} ng/ml | Pred: {pred_conc:6.4f} ng/ml | Diff: {abs(true_conc - pred_conc):.4f}")
