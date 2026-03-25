import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import re

class FullSpectrumDataset(Dataset):
    def __init__(self, data_path, phase='Ag'):
        # 读取数据 (支持 CSV 和 Excel)
        if data_path.endswith('.xlsx'):
            df = pd.read_excel(data_path)
        else:
            df = pd.read_csv(data_path)
            
        self.wavelengths = df['Wavelength'].values
        
        self.spectra = []
        self.concentrations = []
        
        # 解析列名
        for col in df.columns:
            if col == 'Wavelength': continue
            if phase not in col: continue # 只提取指定阶段 (如 'Ag')
            
            # 使用正则提取浓度数值，例如 "0.5ng/ml-Ag-01_07" -> 0.5
            match = re.search(r'([0-9.]+)\s*ng/ml', col)
            if match:
                conc = float(match.group(1))
                # 将光谱数据转为张量
                spec_tensor = torch.FloatTensor(df[col].values)
                self.spectra.append(spec_tensor)
                self.concentrations.append(conc)
                
        # 转换为张量集
        # 增加一个 channel 维度以适应 1D-CNN: shape [N, 1, seq_len]
        self.spectra = torch.stack(self.spectra).unsqueeze(1) 
        
        # 归一化光谱数据到 [0, 1] 区间
        # 优化：从“全局归一”改为“单条归一 (Per-Spectrum MinMax)”，
        # 消除不同环境基线整体漂移对 1D-CNN 浓度的干扰
        # 为了生成器可以反归一化，我们保留全局最大最小值作为缩放基准
        self.spec_min = self.spectra.min()
        self.spec_max = self.spectra.max()
        
        # 逐条归一化
        b, c, l = self.spectra.shape
        spectra_flat = self.spectra.view(b, -1)
        mins = spectra_flat.min(dim=1, keepdim=True)[0]
        maxs = spectra_flat.max(dim=1, keepdim=True)[0]
        self.spectra = ((spectra_flat - mins) / (maxs - mins + 1e-8)).view(b, c, l)
        
        # 浓度使用对数化处理，利于深度学习收敛
        self.concentrations = torch.FloatTensor(np.log10(np.array(self.concentrations) + 1e-3)).unsqueeze(1)
        
        print(f"✅ 成功加载 {len(self.spectra)} 条 {phase} 阶段全光谱数据！波长点数: {len(self.wavelengths)}")

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        return self.spectra[idx], self.concentrations[idx]
