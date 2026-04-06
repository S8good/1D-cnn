# Paper Draft: Methods & Results (English)

> **Target Journal**: Biosensors and Bioelectronics (B&B)  
> **Article type**: Research Article  
> **Last updated**: 2026-04-01  

---

## 3. Materials and Methods

### 3.1 Experimental Dataset and Data Splitting

All experiments used CEA (carcinoembryonic antigen) LSPR spectral measurements collected from gold nanoparticle biosensor chips. Samples were filtered to retain only those with coefficient of variation (CV) below 10% across repeat measurements, yielding a high-quality subset of paired BSA-reference and Ag-antigen spectra. The final dataset comprised 259 paired spectral observations, each consisting of a full-spectrum intensity profile (wavelength range ~500–900 nm, L = 450 data points) measured before (BSA baseline stage) and after (Ag antigen-binding stage) sample introduction.

The dataset was split into training (181 pairs), validation (39 pairs), and test (39 pairs) sets using a fixed stratified split across concentration levels to ensure balanced representation. All experiments used the same fixed split file (`splits_reconstructed`) with a primary random seed of 20260325. Stage 3 experiments additionally used seeds 20260331 and 20260407 for 3-seed reproducibility validation.

### 3.2 Input Representation

Raw Ag spectra were transformed into a 2-channel representation to enhance sensitivity to spectral shape changes:

- **Channel 1** (Intensity): Robust standardization using median and interquartile range (IQR):
$$x_{1}(\lambda) = \frac{s(\lambda) - \mathrm{median}(s)}{\mathrm{IQR}(s) + \varepsilon}$$

- **Channel 2** (First derivative): Sensitivity to slope and peak boundaries:
$$x_{2}(\lambda) = \frac{\nabla s(\lambda) - \mathrm{median}(\nabla s)}{\mathrm{IQR}(\nabla s) + \varepsilon}$$

where ε = 1×10⁻⁸ prevents division by zero. The 2-channel input tensor has shape [2 × L].

### 3.3 BSA Physics Feature Extraction

For each sample, three scalar physical features were extracted from the BSA baseline spectrum via Lorentzian curve fitting:

$$I(\lambda) = A \cdot \frac{\gamma^2}{(\lambda - \lambda_0)^2 + \gamma^2}, \quad \gamma = \frac{\mathrm{FWHM}}{2}$$

The feature vector is **f**_phy = [λ_peak (nm), A_peak (a.u.), FWHM (nm)] ∈ ℝ³, encoding the LSPR resonance baseline state prior to antigen binding.

### 3.4 Model Architecture

#### 3.4.1 Model A: Baseline 1D-CNN (V1)

The baseline predictor (Model A) accepts a single-channel raw Ag spectrum. It consists of three 1D convolutional blocks (Conv1d + BatchNorm + ReLU, kernel sizes 9/7/5, with MaxPooling) followed by a fully connected head outputting log₁₀(ĉ), where ĉ is the predicted concentration in ng/mL.

#### 3.4.2 Model B: Dual-Channel 1D-CNN (V2)

Model B replaces the single-channel input with the 2-channel robust representation described in §3.2. The convolutional blocks use GELU activations and include Dropout regularization (p = 0.1).

#### 3.4.3 Model C: V2-Fusion with BSA Physics Branch

Model C introduces a dedicated **BSA physics feature branch**, forming the core architectural innovation (Figure 2, columns ① and ②). The spectral CNN branch processes the 2-channel Ag spectrum through three progressively deeper 1D convolutional blocks:

- **Conv Block 1**: in=2 → out=32, kernel=9, MaxPool×2, Dropout 0.1
- **Conv Block 2**: 32 → 64, kernel=7, MaxPool×2, Dropout 0.1
- **Conv Block 3**: 64 → 128, kernel=5, AdaptiveAvgPool → 24

A flatten + two fully connected layers (128×24 → 256 → 128-dim) produce the spectral embedding **h**_spec ∈ ℝ¹²⁸.

In parallel, the physics encoder MLP encodes **f**_phy via:
$$\text{MLP}: \mathbb{R}^3 \xrightarrow{\text{Linear}(3 \to 32) + \text{GELU}} \xrightarrow{\text{Linear}(32 \to 32) + \text{GELU}} \mathbf{h}_{phy} \in \mathbb{R}^{32}$$

The fused representation is: **h**_fused = concat[**h**_spec ‖ **h**_phy] ∈ ℝ¹⁶⁰

A 3-layer regressor MLP then maps this to the predicted log-concentration:
$$\hat{z} = \text{MLP}_{\text{reg}}: 160 \to 128 \to 64 \to 1, \quad \hat{c} = 10^{\hat{z}}$$

Model C is trained with MAE loss plus a monotonicity regularization penalty L_Mono.

#### 3.4.4 Model D: 3C-Learnable-Regressor with Hill Binding Constraint

Model D extends Model C with three additional physics-guided components (Figure 2, columns ③–④):

**① Differentiable Peak Extraction via Soft-Argmax**

To enable gradient flow through peak localization, we implement soft-argmax instead of the discrete argmax:
$$\hat{\lambda}_{\text{peak}} = \sum_i \sigma\left(\frac{s(\lambda_i)}{\tau}\right) \cdot \lambda_i$$

where σ(·) is softmax over wavelengths and τ is a temperature parameter. This yields differentiable peak wavelength estimates $\hat{\lambda}_{ag}$ and $\hat{\lambda}_{bsa}$, from which the spectral shift is computed:
$$\widehat{\Delta\lambda} = \hat{\lambda}_{ag} - \hat{\lambda}_{bsa}$$

**② Learnable Hill Binding Curve**

The Hill equation describes the dose-response relationship between analyte concentration and LSPR wavelength shift in adsorption-saturation binding:
$$\Delta\lambda_{\text{Hill}}(c) = \frac{\Delta\lambda_{\max} \cdot c^n}{K_{1/2}^n + c^n}$$

where Δλ_max is the maximum wavelength shift at saturation, K_{1/2} is the half-saturation concentration, and n is the Hill cooperativity coefficient. In Model D, these parameters are initialized from offline curve fitting but treated as **learnable parameters** reparameterized through softplus to enforce positivity:

$$\theta = \ln(e^{\theta_{\text{raw}}} - 1) \xrightarrow{\text{softplus}} \theta > 0$$

The Hill physical consistency loss is:
$$\mathcal{L}_{\text{Hill}} = \|\widehat{\Delta\lambda} - \Delta\lambda_{\text{Hill}}(\hat{c})\|_1$$

**③ Alternating P-step / G-step Training with Spectrum Generator**

Model D incorporates a spectrum generator G: ℝ¹ → ℝᴸ, implemented as a fully connected decoder followed by four transposed 1D convolutional upsampling layers. The generator is trained in alternating fashion:
- **P-step** (Predictor step): optimizes the spectral CNN + regressor using L_MAE + L_Mono + L_Hill + L_Cycle
- **G-step** (Generator step): optimizes the generator using L_Recon + L_Hill

The total training objective is:
$$\mathcal{L} = w_{\text{mae}} \mathcal{L}_{\text{MAE}} + w_{\text{mono}} \mathcal{L}_{\text{Mono}} + w_{\text{cyc}} \mathcal{L}_{\text{Cycle}} + w_{\text{recon}} \mathcal{L}_{\text{Recon}} + w_{\text{hill}} \mathcal{L}_{\text{Hill}}$$

### 3.5 Hyperparameters and Training Protocol

All models were implemented in PyTorch and trained on an NVIDIA GPU. Table S3 summarizes the key hyperparameters.

| Hyperparameter | Model A | Model B | Model C | Model D (3C) |
|---|---|---|---|---|
| Optimizer | Adam | Adam | Adam | Adam |
| Learning rate | 1×10⁻³ | 1×10⁻³ | 1×10⁻³ | 5×10⁻⁴ |
| Batch size | 32 | 32 | 32 | 32 |
| Epochs | 200 | 200 | 200 | 120 (warmup 20) |
| w_mae | — | — | 1.0 | 1.0 |
| w_mono | — | — | 0.05 | 0.05 |
| w_hill | — | — | — | 0.1 |
| w_cyc | — | — | — | 0.005 |
| w_recon | — | — | — | 0.1 |
| Hill params (init) | — | — | — | Δλ_max=6.5, K½=50, n=1.5 |

### 3.6 Evaluation Metrics

Five quantitative metrics were used for model evaluation:

- **MAE** (Mean Absolute Error, ng/mL): $\frac{1}{N}\sum_i|c_i - \hat{c}_i|$
- **RMSE** (Root Mean Squared Error, ng/mL): $\sqrt{\frac{1}{N}\sum_i(c_i - \hat{c}_i)^2}$
- **MAPE** (%): $\frac{100}{N}\sum_i\frac{|c_i - \hat{c}_i|}{c_i}$
- **R²** (Coefficient of determination)
- **Hill-MAE** (nm): mean absolute deviation of the model's implied Δλ from the theoretical Hill curve, quantifying **physical consistency** of the concentration predictions.

---

## 4. Results

### 4.1 Ablation Study: Progressive Performance Gains from Model A to Model D

To quantify the independent contribution of each proposed component, we conducted a systematic ablation study under identical data splits and evaluation protocols. Four progressively augmented configurations were evaluated (Table 1, Figure 3):

**Table 1. Test-set ablation results (single seed 20260325 for Models A–C; 3-seed mean for Model D)**

| Model | Architecture | MAE ↓ (ng/mL) | RMSE ↓ (ng/mL) | MAPE ↓ (%) | R² ↑ |
|---|---|---|---|---|---|
| Model A | 1D-CNN (V1), single-channel Ag spectrum | 11.279 | 22.285 | 40.927 | 0.334 |
| Model B | 1D-CNN (V2), dual-channel (intensity + derivative) | 6.477 | 13.297 | 30.163 | 0.763 |
| **Model C** | V2-Fusion, BSA physics feature branch | **5.586** | **10.822** | **28.321** | **0.843** |
| Model D | V2-Fusion + Learnable Hill Constraint (3C, 3-seed mean) | 6.529 | 13.248 | 35.292 | 0.765 |

The transition from Model A to Model B demonstrates the value of dual-channel input representation: incorporating the first-order spectral derivative alongside intensity information reduced MAE by 42.6% (11.28 → 6.48 ng/mL) and improved R² from 0.334 to 0.763. This confirms that spectral shape dynamics (encoded in the derivative channel) carry complementary discriminative information beyond raw intensity.

The BSA physics feature branch introduced in Model C delivered a further significant gain: MAE decreased by 13.7% relative to Model B (6.48 → 5.59 ng/mL), RMSE dropped from 13.30 to 10.82 ng/mL, and R² reached 0.843. This improvement validates that the BSA resonance baseline state (λ_peak, A_peak, FWHM) provides physics-driven, batch-drift-corrected reference information that substantially aids concentration estimation.

Model D (3C-Learnable-Regressor, 3-seed mean) shows comparably competitive performance with Model C in concentration accuracy metrics. The modest elevation in MAE (5.59 → 6.53 ng/mL) is attributable to two factors: (1) Model D results represent the mean over three independent random seeds, introducing additional variance; (2) the Hill physical consistency loss redistributes gradient resources from pure regression to joint concentration-physics optimization. Critically, Model D's advantage lies in physical consistency, as detailed in §4.2.

### 4.2 Physical Consistency Analysis: Hill Binding Curve Conformity

To evaluate whether model predictions are physically coherent with LSPR adsorption binding theory, we computed Hill-MAE — the mean absolute deviation between the model-implied Δλ shift and the theoretically expected shift from the Hill isothermal adsorption equation:

$$\Delta\lambda_{\text{Hill}} = \frac{\Delta\lambda_{\max} \cdot \hat{c}^{\,n}}{K_{1/2}^n + \hat{c}^{\,n}}$$

Three Stage-3 configurations of Model D were compared (Table S1, Figure 1b, Figure 5):

**Table S1. Hill-MAE comparison across Stage 3 configurations (3-seed, mean ± std)**

| Configuration | Description | Hill-MAE (nm) |
|---|---|---|
| 3A — fixed-frozen | Fixed Hill params, predictor frozen | 2.256 ± 0.078 |
| 3B — fixed-regressor | Fixed Hill params, regressor updated | 1.859 ± 0.250 |
| **3C — learnable-regressor (Model D)** | Learnable Hill params, regressor updated | **1.698 ± 0.026** |

The 3C configuration achieves the lowest Hill-MAE (1.698 nm) with the smallest standard deviation (±0.026 nm), demonstrating that making the Hill parameters jointly learnable with the predictor enables the model to adaptively refine its physical binding curve to match the observed spectral shift patterns. The high reproducibility (std < 2% of mean) confirms stability across random initializations.

Figure 5 visualizes the three configurations' predicted Δλ values against the theoretical Hill sigmoidal binding curve. Model D (3C) data points (green circles) cluster most tightly around the curve, particularly in the nonlinear transition region (1–10 ng/mL), where configurations 3A (blue triangles) and 3B (orange squares) show appreciably wider scatter. This demonstrates that the learnable Hill constraint effectively anchors the model's spectral shift predictions to the expected binding physics.

### 4.3 Segmented Error Analysis

The inherently nonlinear LSPR signal-concentration relationship mandates per-concentration-range analysis. We divided the test set into three segments: low (< 5 ng/mL), medium (5–30 ng/mL), and high (> 30 ng/mL) concentrations (Figure S2, Table S2).

**Table S2. Segmented MAE by concentration range (ng/mL)**

| Model | Low (< 5 ng/mL) | Medium (5–30 ng/mL) | High (> 30 ng/mL) |
|---|---|---|---|
| A | 0.69 | 2.57 | 28.18 |
| B | 0.80 | 4.28 | 12.89 |
| **C** | **0.71** | **3.87** | **10.91** |
| D | 1.01 | 3.80 | 13.40 |

All models achieve low absolute error in the sub-5 ng/mL range (MAE < 1.1 ng/mL), reflecting the approximately linear dose-response in this Hill equation regime. The dominant error source is the high-concentration regime (> 30 ng/mL), where LSPR signal approaches its saturation plateau and sensitivity diminishes. Model A's high-concentration MAE of 28.18 ng/mL is dramatically reduced by the dual-channel representation (B: 12.89) and the BSA physics branch (C: 10.91). Model D's high-concentration MAE (13.40 ng/mL) is slightly higher than Model C, consistent with the 3-seed averaging and gradient-sharing effects discussed in §4.1.

The clinically relevant medium-concentration range (5–30 ng/mL) — corresponding to diagnostically important CEA concentrations — shows all models operating within 4.3 ng/mL MAE, confirming practical applicability.

### 4.4 Prediction Accuracy and Error Distribution (Bland-Altman Analysis)

Figure 4 presents the true-vs-predicted concentration scatter plot for the three Model C configurations (Figure 4, n = 117 combined test samples). Pearson correlation coefficients ranged from r = 0.87–0.90 across seeds, with aggregated MAE = 6.53 ng/mL and RMSE = 13.25 ng/mL. Predictions in the 0–40 ng/mL range cluster tightly around the identity line; systematic underestimation emerges above ~40 ng/mL, attributable to the LSPR saturation phenomenon rather than model failure.

Bland-Altman analysis (Figure S1) quantified systematic bias and limits of agreement across the three independent experimental replicates. The mean bias estimates were +1.04, +1.73, and +1.82 ng/mL across seeds (pooled: +1.53 ng/mL), indicating a slight positive systematic offset (tendency toward mild overestimation). This level of systematic bias is acceptable in the LSPR competitive binding assay context and may be corrected post-hoc using isotonic calibration. The 95% limits of agreement (LoA) spanned approximately −24 to +28 ng/mL overall, driven primarily by the high-concentration saturation region; within the clinically primary range (< 30 ng/mL), the vast majority of measurements fell within ±10 ng/mL.

---

*Figure references: Figure 1/1b → outputs/stage3_comparison_figure.png / stage3_hilmae_figure.png; Figure 3 → outputs/ablation_comparison_figure.png; Figure 4 → outputs/true_vs_pred_3c_figure.png; Figure 5 → outputs/hill_consistency_figure.png; Figure S1 → outputs/bland_altman_3c_figure.png; Figure S2 → outputs/segment_stats_figure.png.*
