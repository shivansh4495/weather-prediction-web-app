#!/usr/bin/env python3
"""
Weather + AQI Dual-Model Training Script
=========================================
Trains CNN-LSTM models for both Temperature AND AQI prediction.
Optimized for RTX 5050 (CUDA 12.x) + Ryzen 7 with mixed-precision training.

Usage:
    python train_weather_aqi.py

Outputs:
    best_model.pth      - Temperature forecasting model
    best_aqi_model.pth  - AQI forecasting model
    scaler.pkl          - Feature scaler
    aqi_scaler.pkl      - AQI-specific scaler
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. Device Setup  — RTX 5050 (sm_120 / CUDA 12.8)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Weather + AQI Dual-Model Training")
print("=" * 60)
print(f"PyTorch  : {torch.__version__}")
print(f"CUDA ok  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    device = torch.device('cuda')
    USE_AMP = True          # mixed-precision for RTX 50xx
else:
    print("WARNING  : GPU not found, falling back to CPU")
    device = torch.device('cpu')
    USE_AMP = False
print()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Data Loading
# ─────────────────────────────────────────────────────────────────────────────
FILE = "Filtered_Weather_Forecasting_Datasetls.xlsx"
if not os.path.exists(FILE):
    raise FileNotFoundError(f"{FILE} not found. Place it in the same directory.")

print("Loading dataset …")
df = pd.read_excel(FILE)
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

# ── Column normalisation ──────────────────────────────────────────────────────
col_map = {
    'temperature_2m (°C)': 'temperature',
    'rain (mm)': 'rain',
    'rain (...)': 'rain',
    'relative_humidity_2m (%)': 'humidity',
    'pressure_msl (hPa)': 'pressure',
    'cloud_cover (%)': 'cloud_cover',
    'wind_speed_10m (km/h)': 'wind_speed',
    'wind_direction_10m (A°)': 'wind_dir',
}
df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

# Temporal features
df['hour']  = df['time'].dt.hour
df['dow']   = df['time'].dt.dayofweek
df['month'] = df['time'].dt.month
df['sin_h'] = np.sin(2 * np.pi * df['hour'] / 24)
df['cos_h'] = np.cos(2 * np.pi * df['hour'] / 24)
df['sin_m'] = np.sin(2 * np.pi * df['month'] / 12)
df['cos_m'] = np.cos(2 * np.pi * df['month'] / 12)

# Required features
FEAT_COLS = ['temperature', 'humidity', 'pressure', 'rain',
             'cloud_cover', 'wind_speed', 'sin_h', 'cos_h', 'sin_m', 'cos_m']
FEAT_COLS = [c for c in FEAT_COLS if c in df.columns]

HAS_AQI = 'AQI' in df.columns
if HAS_AQI:
    df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')
    df['AQI'] = df['AQI'].fillna(df['AQI'].median())
    df['AQI'] = np.clip(df['AQI'], 20, 500)
    print(f"AQI column found  — range [{df['AQI'].min():.0f}, {df['AQI'].max():.0f}]")
else:
    print("WARNING: AQI column not found — will build formula-based AQI")

df = df[FEAT_COLS + (['AQI'] if HAS_AQI else [])].dropna()
print(f"Dataset ready     — {len(df):,} rows × {len(FEAT_COLS)} features")
print(f"Features          : {FEAT_COLS}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Model Architecture
# ─────────────────────────────────────────────────────────────────────────────
class WeatherLSTM(nn.Module):
    """
    CNN-LSTM hybrid — same architecture as the original CompactForecastModel
    but slightly wider for multi-feature input.
    """
    def __init__(self, n_features: int, dropout: float = 0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64,      kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 128,     kernel_size=3, padding=1), nn.ReLU(),
        )
        self.lstm = nn.LSTM(128, 128, num_layers=2,
                            batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(128, 1)

    def forward(self, x):                       # x: (B, T, F)
        x = x.permute(0, 2, 1)                 # (B, F, T)
        x = self.cnn(x)                        # (B, 128, T)
        x = x.permute(0, 2, 1)                 # (B, T, 128)
        x, _ = self.lstm(x)
        return self.fc(self.drop(x))           # (B, T, 1)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
SEQ_LEN  = 168   # 7 days input
PRED_LEN = 168   # 7 days output
BATCH    = 64
EPOCHS   = 120
PATIENCE = 20

def make_sequences(data_np, target_np, stride=1):
    X, y = [], []
    for i in range(0, len(data_np) - SEQ_LEN - PRED_LEN + 1, stride):
        X.append(data_np[i : i + SEQ_LEN])
        y.append(target_np[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)[..., None]

def train_model(model, train_dl, val_dl, tag):
    opt      = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit     = nn.HuberLoss()
    amp_scl  = GradScaler() if USE_AMP else None

    best_val = float('inf')
    best_wts = None
    patience_cnt = 0

    for epoch in range(1, EPOCHS + 1):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        t_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            if USE_AMP:
                with autocast():
                    pred = model(xb)
                    loss = crit(pred, yb)
                amp_scl.scale(loss).backward()
                amp_scl.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                amp_scl.step(opt)
                amp_scl.update()
            else:
                pred = model(xb)
                loss = crit(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            t_loss += loss.item()

        # ── validate ───────────────────────────────────────────────────────
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                if USE_AMP:
                    with autocast():
                        pred = model(xb)
                        v_loss += crit(pred, yb).item()
                else:
                    v_loss += crit(model(xb), yb).item()

        avg_t = t_loss / len(train_dl)
        avg_v = v_loss / len(val_dl)
        sched.step()

        if epoch % 10 == 0 or epoch <= 5:
            lr = opt.param_groups[0]['lr']
            print(f"  [{tag}] Epoch {epoch:3d}  train={avg_t:.5f}  val={avg_v:.5f}  lr={lr:.2e}")

        if avg_v < best_val:
            best_val = avg_v
            best_wts = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  [{tag}] Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_wts)
    return model, best_val

# ─────────────────────────────────────────────────────────────────────────────
# 5. Train Temperature Model
# ─────────────────────────────────────────────────────────────────────────────
print("─" * 60)
print("  [1/2]  Temperature Model")
print("─" * 60)

feat_scaler = StandardScaler()
feat_scaled = feat_scaler.fit_transform(df[FEAT_COLS])

temp_idx    = FEAT_COLS.index('temperature')
temp_scaler = StandardScaler()
temp_scaled = temp_scaler.fit_transform(df[['temperature']]).flatten()

X_t, y_t = make_sequences(feat_scaled, temp_scaled, stride=4)
split     = int(0.8 * len(X_t))
X_tr, X_val = X_t[:split], X_t[split:]
y_tr, y_val = y_t[:split], y_t[split:]

tr_dl  = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), BATCH, shuffle=True,  num_workers=0, pin_memory=device.type=='cuda')
val_dl = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), BATCH, shuffle=False, num_workers=0, pin_memory=device.type=='cuda')

temp_model = WeatherLSTM(len(FEAT_COLS)).to(device)
print(f"  Params: {sum(p.numel() for p in temp_model.parameters()):,}")
temp_model, best_t = train_model(temp_model, tr_dl, val_dl, "TEMP")
print(f"  Best val loss: {best_t:.5f}\n")

torch.save(temp_model.state_dict(), "best_model.pth")
joblib.dump(feat_scaler, "scaler.pkl")
joblib.dump(temp_scaler, "temp_scaler.pkl")
print("  Saved → best_model.pth / scaler.pkl / temp_scaler.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Train AQI Model
# ─────────────────────────────────────────────────────────────────────────────
print()
print("─" * 60)
print("  [2/2]  AQI Model")
print("─" * 60)

if HAS_AQI:
    aqi_scaler = StandardScaler()
    aqi_scaled = aqi_scaler.fit_transform(df[['AQI']]).flatten()
    joblib.dump(aqi_scaler, "aqi_scaler.pkl")
else:
    # Build a synthetic AQI based on formula for training signal
    temp_v  = df['temperature'].values
    hum_v   = df.get('humidity', pd.Series([60]*len(df))).values
    wind_v  = df.get('wind_speed', pd.Series([10]*len(df))).values
    syn_aqi = np.clip(40 + 1.7*temp_v + 0.35*hum_v - 1.1*wind_v, 20, 500)
    aqi_scaler = StandardScaler()
    aqi_scaled = aqi_scaler.fit_transform(syn_aqi.reshape(-1,1)).flatten()
    joblib.dump(aqi_scaler, "aqi_scaler.pkl")

X_a, y_a = make_sequences(feat_scaled, aqi_scaled, stride=4)
split     = int(0.8 * len(X_a))
X_atr, X_aval = X_a[:split], X_a[split:]
y_atr, y_aval = y_a[:split], y_a[split:]

atr_dl  = DataLoader(TensorDataset(torch.tensor(X_atr), torch.tensor(y_atr)), BATCH, shuffle=True,  num_workers=0, pin_memory=device.type=='cuda')
avl_dl  = DataLoader(TensorDataset(torch.tensor(X_aval), torch.tensor(y_aval)), BATCH, shuffle=False, num_workers=0, pin_memory=device.type=='cuda')

aqi_model = WeatherLSTM(len(FEAT_COLS)).to(device)
print(f"  Params: {sum(p.numel() for p in aqi_model.parameters()):,}")
aqi_model, best_a = train_model(aqi_model, atr_dl, avl_dl, "AQI ")
print(f"  Best val loss: {best_a:.5f}\n")

torch.save(aqi_model.state_dict(), "best_aqi_model.pth")
print("  Saved → best_aqi_model.pth / aqi_scaler.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Quick Evaluation
# ─────────────────────────────────────────────────────────────────────────────
print()
print("─" * 60)
print("  Evaluation on validation sets")
print("─" * 60)

def evaluate(model, scaler_out, val_dl_ev, label):
    model.eval()
    preds, acts = [], []
    with torch.no_grad():
        for xb, yb in val_dl_ev:
            xb = xb.to(device)
            if USE_AMP:
                with autocast():
                    p = model(xb).cpu().numpy()
            else:
                p = model(xb).cpu().numpy()
            preds.append(p); acts.append(yb.numpy())
    preds = np.concatenate(preds).reshape(-1, 1)
    acts  = np.concatenate(acts).reshape(-1, 1)
    p_inv = scaler_out.inverse_transform(preds).flatten()
    a_inv = scaler_out.inverse_transform(acts).flatten()
    mae   = mean_absolute_error(a_inv, p_inv)
    rmse  = mean_squared_error(a_inv, p_inv) ** 0.5
    r2    = r2_score(a_inv, p_inv)
    print(f"  {label:10s}  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}")
    return mae

mae_t = evaluate(temp_model, temp_scaler, val_dl,  "TEMPERATURE")
if HAS_AQI:
    evaluate(aqi_model, aqi_scaler, avl_dl, "AQI")

print()
if mae_t < 2.0:
    print("  ✅  Temperature MAE < 2 °C  — excellent!")
elif mae_t < 3.5:
    print("  ✅  Temperature MAE < 3.5 °C — good")
else:
    print("  ⚠️   Consider more epochs or data augmentation")

print()
print("Training complete! Files produced:")
print("  best_model.pth    — temperature LSTM")
print("  best_aqi_model.pth— AQI LSTM")
print("  scaler.pkl        — feature scaler (shared)")
print("  temp_scaler.pkl   — temperature inverse-transform")
print("  aqi_scaler.pkl    — AQI inverse-transform")
