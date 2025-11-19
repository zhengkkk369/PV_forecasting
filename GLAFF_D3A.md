这份文档汇总了我们讨论的核心策略与架构设计，旨在指导 AI Agent 生成一个**高鲁棒性、低成本的在线时间序列预测系统**。该系统集成了 **GLAFF**（作为抗噪与归一化插件）和 **D$^3$A**（作为显式漂移检测与触发器）。

---

# Project Spec: GLAFF-D$^3$A-Lite 
**Robust Adaptive Online Time Series Forecasting Framework**

## 1. 系统架构概述 (System Architecture)

本系统采用 **"GLAFF-Inside, D$^3$A-Outside"** 的设计模式，旨在结合 GLAFF 的鲁棒性与 D$^3$A 的适应性，同时将训练成本降至最低。

### 核心组件
1.  **Backbone Model (宿主模型)**: 任意主流时序预测模型 (e.g., iTransformer, TimesNet, DLinear)。
2.  **GLAFF Plugin (增强插件)**: 
    *   负责处理输入数据的全局时间戳依赖。
    *   **关键职责**: 替代 D$^3$A 的高斯噪声增强，通过 `Robust Denormalization` (Median/IQR) 强制对齐历史数据与当前数据的分布。
3.  **Drift Detector (D$^3$A Lite)**:
    *   **关键职责**: 仅作为触发器 (Trigger)。监控预测误差的 Z-score，判断是否发生 GLAFF 无法处理的结构性漂移。
4.  **Adaptation Manager (适应管理器)**:
    *   **关键策略**: **"冻结躯干，微调末梢" (Freeze Trunk, Tune Head)**。
    *   当漂移触发时，仅更新 Backbone 的输出层 (Head) 和 GLAFF 的组合器 (Combiner)。

---

## 2. 关键逻辑流程 (Workflow)

### Step 1: 前向推理 (Inference)
*   输入当前窗口 $X_t$ 和时间戳 $T_t$。
*   GLAFF 进行鲁棒归一化，Backbone 提取特征。
*   GLAFF 的 Adaptive Combiner 融合全局与局部结果，输出 $\hat{Y}_t$。

### Step 2: 漂移检测 (Detection)
*   接收真实标签 $Y_t$ (或延迟反馈)。
*   计算 Loss $L_t = \| \hat{Y}_t - Y_t \|$。
*   **D$^3$A Monitor**:
    *   维护 Loss 的滑动窗口。
    *   计算当前 Loss 的 Z-score: $Z = \frac{L_t - \mu_{loss}}{\sigma_{loss}}$。
    *   若 $Z > Threshold$ 且 GLAFF 无法抑制误差 $\rightarrow$ **触发 Alarm**。

### Step 3: 极简适应 (Lite Adaptation)
*   **条件**: 仅当 Alarm 触发时执行。
*   **数据**: 从 Memory Bank 提取最近的历史样本 $(X_{history}, Y_{history})$。
*   **预处理**: **不做** D$^3$A 原生的噪声注入 (Gap Filling)，直接依赖 GLAFF 的内部归一化机制处理分布差异。
*   **参数更新**:
    *   `Backbone.Encoder`: **Frozen (冻结)**
    *   `GLAFF.Mapper`: **Frozen (冻结)** (可选：视情况解冻)
    *   `Backbone.Projection_Head`: **Active (更新)**
    *   `GLAFF.Combiner`: **Active (更新)**
*   **重置**: 清空 Detector 的统计窗口，适应新概念。

---

## 3. 核心代码规范 (Pseudo-code for Agent)

请基于以下伪代码结构生成完整的 Python 代码。

### 3.1 GLAFF Plugin (Standard Implementation)
*保留原有的 Attention-based Mapper 和 Adaptive Combiner，确保 `Robust Denormalization` 逻辑正确。*

```python
import torch
from torch import nn

class GLAFFPlugin(nn.Module):
    def __init__(self, args, channel):
        super(GLAFFPlugin, self).__init__()
        # ... (Standard Initialization as per provided code) ...
        # self.Encoder = ...
        # self.MLP = ... (The Adaptive Combiner)

    def forward(self, x_enc_true, x_mark_enc, x_dec_pred, x_mark_dec):
        # 1. Standardize inputs (Mean/Std)
        # ...
        
        # 2. Global Mapping (Attention Mapper)
        # x_enc_map = self.Encoder(x_mark_enc)
        
        # 3. Robust Denormalization (Crucial for Alignment)
        # Using Median and Quantile (IQR) to align distribution
        # robust_means_map = torch.median(...)
        # robust_stdev_map = torch.quantile(...)
        # x_enc_map = (x_enc_map - robust_means_map) / ...
        
        # 4. Adaptive Combination (The part to be updated during drift)
        # weight = self.MLP(...)
        # pred = ...
        
        return pred
```

### 3.2 Adaptive Model Wrapper (集成控制核心)
*负责封装 Backbone 和 GLAFF，并提供“参数冻结/解冻”的接口。*

```python
class AdaptiveModel(nn.Module):
    def __init__(self, backbone, plugin_args):
        super().__init__()
        self.backbone = backbone
        self.plugin = GLAFFPlugin(plugin_args, backbone.enc_in)
        
    def forward(self, x, x_mark, ...):
        # 1. Get Backbone prediction (Local)
        # Assume backbone returns un-normalized local prediction
        dec_out = self.backbone(x, x_mark, ...) 
        
        # 2. Apply GLAFF (Global + Robustness)
        # Pass the backbone output to plugin for refinement
        final_pred = self.plugin(x, x_mark, dec_out, ...)
        return final_pred

    def toggle_adaptation_mode(self, enable=True):
        """
        Switch between 'Frozen Trunk' and 'Full Training' modes.
        Strategy: Freeze Encoder, Update Head & Combiner.
        """
        # 1. Default: Freeze everything
        for param in self.parameters():
            param.requires_grad = not enable 

        if enable:
            # 2. Unfreeze Backbone Projection Head (Output Layer)
            # Note: Adjust 'projection' to match the specific backbone's attribute name
            if hasattr(self.backbone, 'projection'):
                for param in self.backbone.projection.parameters():
                    param.requires_grad = True
            elif hasattr(self.backbone, 'head'): # Common in some implementations
                for param in self.backbone.head.parameters():
                    param.requires_grad = True
            
            # 3. Unfreeze GLAFF Adaptive Combiner
            for param in self.plugin.MLP.parameters():
                param.requires_grad = True
            
            # Optional: Keep GLAFF Encoder frozen to assume global periodicity implies 
            # stable timestamps, unless drift is extremely severe.
```

### 3.3 Drift Detector (D$^3$A Logic)
*显式检测器，监控 GLAFF 处理后的 Loss。*

```python
import numpy as np
from collections import deque

class ConceptDriftDetector:
    def __init__(self, window_size=32, threshold=3.0):
        self.window_size = window_size
        self.threshold = threshold
        self.loss_history = deque(maxlen=window_size)
        self.zm_stats = deque(maxlen=window_size * 10) # Longer history for stable stats
        
    def update_and_check(self, current_loss):
        self.loss_history.append(current_loss)
        
        # Need enough data to establish a baseline
        if len(self.zm_stats) < self.window_size:
            self.zm_stats.append(current_loss)
            return False
            
        # Calculate stats from reference history (not the immediate recent window)
        mu = np.mean(self.zm_stats)
        std = np.std(self.zm_stats) + 1e-6
        
        # Calculate Z-score of the current short-term window
        current_window_mean = np.mean(self.loss_history)
        z_score = (current_window_mean - mu) / (std / np.sqrt(len(self.loss_history)))
        
        drift_detected = False
        if z_score > self.threshold:
            drift_detected = True
        else:
            # If no drift, update reference stats slowly
            self.zm_stats.append(current_loss)
            
        return drift_detected

    def reset_on_drift(self):
        self.loss_history.clear()
        # Optional: Clear stats or keep partial history
        self.zm_stats.clear() 
```

### 3.4 Main Online Loop (Manager)
*串联整个流程，实现低成本适应。*

```python
class OnlineForecaster:
    def __init__(self, model, detector, memory_bank, optimizer_lr=0.001):
        self.model = model
        self.detector = detector
        self.memory = memory_bank
        self.lr = optimizer_lr
        
    def step(self, x_t, x_mark_t, y_true):
        # --- 1. Inference ---
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x_t, x_mark_t)
            loss = self.criterion(y_pred, y_true)
            
        # --- 2. Detection ---
        is_drift = self.detector.update_and_check(loss.item())
        
        # --- 3. Adaptation (Triggered only on Drift) ---
        if is_drift:
            print(f"Drift detected! Loss: {loss.item():.4f}. Adapting...")
            
            # A. Configure Model for Lite Training
            self.model.toggle_adaptation_mode(enable=True)
            
            # B. Create Optimizer for only active parameters
            # Filtering reduces memory and computation
            active_params = filter(lambda p: p.requires_grad, self.model.parameters())
            optimizer = torch.optim.Adam(active_params, lr=self.lr)
            
            # C. Replay Training (No D3A Noise Injection needed)
            # GLAFF inside the model handles distribution alignment
            samples_x, samples_mark, samples_y = self.memory.get_recent_samples(n=64)
            
            self.model.train()
            for _ in range(self.train_steps): # e.g., 3-5 steps is usually enough for head tuning
                optimizer.zero_grad()
                pred = self.model(samples_x, samples_mark)
                train_loss = self.criterion(pred, samples_y)
                train_loss.backward()
                optimizer.step()
                
            # D. Reset
            self.detector.reset_on_drift()
            self.model.eval() # Switch back to eval
            
        # --- 4. Update Memory ---
        self.memory.add(x_t, x_mark_t, y_true)
        
        return y_pred
```

---

## 4. 总结与优势 (Summary)

*   **互补性**: 利用 GLAFF 处理短期输入噪声，防止 D$^3$A 误报；利用 D$^3$A 处理长期结构性漂移，解决 GLAFF 的局限。
*   **低成本**: 废弃了 D$^3$A 复杂的噪声生成计算，利用 GLAFF 现成的统计量；适应阶段仅更新极少量的参数（Head + Combiner），适合边缘设备或高频交易场景。
*   **实现路径**: 这是一个纯工程化的改进，不需要重新推导数学公式，只需在代码逻辑上进行模块重组。