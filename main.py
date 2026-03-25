import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ==========================================
# 1. 数据加载与对齐 (Data Alignment)
# ==========================================
def load_and_align_data():
    # 获取 main.py 文件所在的绝对路径目录
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 动态拼接出绝对路径
    filtered_path = os.path.join(BASE_DIR, "data", "filtered", "cea_training_data_20pct.csv")
    spectra_path = os.path.join(BASE_DIR, "data", "processed", "Reconstructed_Preprocessed_Spectra.csv")
    
    # 添加错误捕捉，展现专业的工程素养
    if not os.path.exists(filtered_path):
        raise FileNotFoundError(f"【大师提醒】找不到特征数据集，请检查路径: {filtered_path}")
    if not os.path.exists(spectra_path):
        raise FileNotFoundError(f"【大师提醒】找不到全光谱数据集，请检查路径: {spectra_path}")
        
    df_filtered = pd.read_csv(filtered_path)
    df_spectra = pd.read_csv(spectra_path)
    wavelengths = df_spectra['Wavelength'].values
    
    print(f"✅ 成功加载数据！检测到波长范围: {wavelengths.min()}nm - {wavelengths.max()}nm")
    
    return df_filtered, df_spectra, wavelengths

# ==========================================
# 2. 构建模型：残差物理网络 (Residual Physics Net)
# ==========================================
class LSPRResidualNet(nn.Module):
    def __init__(self, input_dim=4, output_dim=2):
        super(LSPRResidualNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim) # 输出 Δlambda 和 ΔA
        )
        
    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. 物理重建函数 (Spectral Reconstruction)
# ==========================================
def lorentzian_reconstruct(wavelengths, peak_pos, amplitude, fwhm):
    """利用物理公式将特征值还原为光谱曲线"""
    gamma = fwhm / 2
    intensity = amplitude * (gamma**2) / ((wavelengths - peak_pos)**2 + gamma**2)
    return intensity

# ==========================================
# 4. 主训练与预测流程
# ==========================================
def main():
    # A. 准备数据
    df_feat, df_spec, wavelengths = load_and_align_data()
    
    # 特征准备：[log浓度, BSA峰位, BSA峰强, BSA半宽]
    X = np.column_stack([
        np.log10(df_feat['c_ng_ml'] + 1e-3),
        df_feat['lambda_peak_nm_pre'],
        df_feat['Apeak_pre'],
        df_feat['fwhm_nm_pre']
    ])
    
    # 目标准备：[Δlambda, ΔA]
    y = np.column_stack([
        df_feat['lambda_peak_nm_post'] - df_feat['lambda_peak_nm_pre'],
        df_feat['Apeak_post'] - df_feat['Apeak_pre']
    ])
    
    # 标准化
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # 转为 PyTorch 张量
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y_scaled)
    
    # B. 训练模型
    model = LSPRResidualNet()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("开始训练...")
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # C. 反向生成验证 (以第一组数据为例)
    model.eval()
    with torch.no_grad():
        test_idx = 0
        # 1. 预测增量
        pred_delta_scaled = model(X_tensor[test_idx].unsqueeze(0))
        pred_delta = scaler_y.inverse_transform(pred_delta_scaled.numpy())[0]
        
        # 2. 计算 Ag 阶段预测特征值
        pred_post_lambda = df_feat.iloc[test_idx]['lambda_peak_nm_pre'] + pred_delta[0]
        pred_post_A = df_feat.iloc[test_idx]['Apeak_pre'] + pred_delta[1]
        
        # 3. 物理重建光谱图
        # 预测的 Ag 光谱
        spec_pred = lorentzian_reconstruct(wavelengths, pred_post_lambda, pred_post_A, df_feat.iloc[test_idx]['fwhm_nm_post'])
        # 实验的 Ag 光谱 (从 df_feat 中的特征值还原作为对比)
        spec_exp = lorentzian_reconstruct(wavelengths, df_feat.iloc[test_idx]['lambda_peak_nm_post'], df_feat.iloc[test_idx]['Apeak_post'], df_feat.iloc[test_idx]['fwhm_nm_post'])
        
        # D. 可视化
        plt.figure(figsize=(10, 6))
        plt.plot(wavelengths, spec_exp, 'r-', label='Experimental Ag (Post)')
        plt.plot(wavelengths, spec_pred, 'b--', label='AI Reconstructed Ag (Post)')
        plt.title(f"Concentration: {df_feat.iloc[test_idx]['c_ng_ml']} ng/ml - Spectral Reconstruction")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Absorbance (a.u.)")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()