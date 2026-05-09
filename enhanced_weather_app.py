#!/usr/bin/env python3


import base64
import io
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flask import Flask, jsonify, render_template

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


try:
    import torch
    import torch.nn as nn
    from torch.cuda.amp import autocast
    TORCH_OK = True
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        GPU_NAME = torch.cuda.get_device_name(0)
        print(f"✅ GPU : {GPU_NAME}")
    else:
        DEVICE = torch.device('cpu')
        GPU_NAME = "CPU"
        print("⚠️  GPU not detected — using CPU")
except Exception as e:
    TORCH_OK = False
    DEVICE = None
    GPU_NAME = "CPU"
    print(f"⚠️  PyTorch unavailable: {e}")

try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False


if TORCH_OK:
    class WeatherLSTM(nn.Module):
        def __init__(self, n_features: int, dropout: float = 0.2):
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv1d(n_features, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv1d(32, 64,      kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv1d(64, 128,     kernel_size=3, padding=1), nn.ReLU(),
            )
            self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True, dropout=dropout)
            self.drop = nn.Dropout(dropout)
            self.fc   = nn.Linear(128, 1)

        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = self.cnn(x)
            x = x.permute(0, 2, 1)
            x, _ = self.lstm(x)
            return self.fc(self.drop(x))


app = Flask(__name__)
app.config['SECRET_KEY'] = 'weather-aqi-secret-2025'

# Global state
historical_data  = None
predictions_data = None
temp_model       = None
aqi_model        = None
feat_scaler      = None
temp_scaler      = None
aqi_scaler       = None
FEAT_COLS        = []
SEQ_LEN          = 168

NOIDA = {'lat': 28.5355, 'lng': 77.3910, 'name': 'Noida, Uttar Pradesh'}


def get_compute_label():
    if TORCH_OK and torch.cuda.is_available():
        return f"GPU ({GPU_NAME})"
    return "CPU"

def classify_aqi(aqi: float) -> str:
    aqi = int(aqi)
    if aqi <= 50:   return 'Good'
    if aqi <= 100:  return 'Moderate'
    if aqi <= 150:  return 'Unhealthy for Sensitive'
    if aqi <= 200:  return 'Unhealthy'
    if aqi <= 300:  return 'Very Unhealthy'
    return 'Hazardous'

def get_season(month: int) -> str:
    return {12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring',  4: 'Spring',  5: 'Spring',
            6: 'Summer',  7: 'Summer',  8: 'Summer'}.get(month, 'Autumn')

def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['hour']  = df['time'].dt.hour
    df['dow']   = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month
    df['sin_h'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_h'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['sin_m'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_m'] = np.cos(2 * np.pi * df['month'] / 12)
    return df


def load_models():
    global temp_model, aqi_model, feat_scaler, temp_scaler, aqi_scaler

    if not (TORCH_OK and JOBLIB_OK):
        print("⚠️  PyTorch/joblib not available — ML inference disabled")
        return

    n = len(FEAT_COLS)
    if n == 0:
        print("⚠️  Feature columns not set — load data first")
        return

    # Feature scaler
    for path in ['scaler.pkl']:
        if Path(path).exists():
            feat_scaler = joblib.load(path)
            print(f"✅ Feature scaler loaded from {path}")
            break

    # Temperature model + scaler
    for mpath, spath in [('best_model.pth', 'temp_scaler.pkl')]:
        if Path(mpath).exists():
            try:
                m = WeatherLSTM(n).to(DEVICE)
                ck = torch.load(mpath, map_location=DEVICE)
                # Handle both raw state-dict and wrapped checkpoint
                sd = ck.get('model_state_dict', ck)
                m.load_state_dict(sd, strict=False)
                m.eval()
                temp_model = m
                print(f"✅ Temperature model loaded ({mpath})")
            except Exception as e:
                print(f"⚠️  Temperature model failed: {e}")

        if Path(spath).exists():
            temp_scaler = joblib.load(spath)
            print(f"✅ Temperature scaler ({spath})")
        elif Path('scaler.pkl').exists() and feat_scaler is not None:
            # fallback: extract from feature scaler if possible
            from sklearn.preprocessing import StandardScaler
            ts = StandardScaler()
            ts.mean_ = np.array([historical_data['temperature'].mean()]) if historical_data is not None else np.array([25.0])
            ts.scale_ = np.array([historical_data['temperature'].std()]) if historical_data is not None else np.array([8.0])
            ts.var_ = ts.scale_ ** 2
            ts.n_features_in_ = 1
            temp_scaler = ts

    
    if Path('best_aqi_model.pth').exists():
        try:
            m = WeatherLSTM(n).to(DEVICE)
            ck = torch.load('best_aqi_model.pth', map_location=DEVICE)
            sd = ck.get('model_state_dict', ck)
            m.load_state_dict(sd, strict=False)
            m.eval()
            aqi_model = m
            print("✅ AQI model loaded")
        except Exception as e:
            print(f"⚠️  AQI model failed: {e}")

    if Path('aqi_scaler.pkl').exists():
        aqi_scaler = joblib.load('aqi_scaler.pkl')
        print("✅ AQI scaler loaded")


def load_historical_data() -> bool:
    global historical_data, FEAT_COLS

    xlsx = 'Filtered_Weather_Forecasting_Datasetls.xlsx'
    if not Path(xlsx).exists():
        print(f"❌ {xlsx} not found")
        return False

    try:
        df = pd.read_excel(xlsx)
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time']).sort_values('time').reset_index(drop=True)

        col_map = {
            'temperature_2m (°C)': 'temperature',
            'rain (mm)': 'rain', 'rain (...)': 'rain',
            'relative_humidity_2m (%)': 'humidity',
            'pressure_msl (hPa)': 'pressure',
            'cloud_cover (%)': 'cloud_cover',
            'wind_speed_10m (km/h)': 'wind_speed',
            'wind_direction_10m (A°)': 'wind_dir',
        }
        df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

        df = build_temporal_features(df)

        defaults = {'rain': 0.0, 'humidity': 60.0, 'pressure': 1012.0,
                    'cloud_cover': 35.0, 'wind_speed': 10.0, 'wind_dir': 180.0}
        for col, val in defaults.items():
            if col not in df.columns:
                df[col] = val
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(val)

        df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
        df = df.dropna(subset=['temperature'])

        # AQI
        if 'AQI' in df.columns:
            df['AQI'] = pd.to_numeric(df['AQI'], errors='coerce')
            df['AQI'] = df['AQI'].fillna(df['AQI'].median())
            df['AQI'] = np.clip(df['AQI'], 20, 500).astype(int)
        else:
            df['AQI'] = _formula_aqi(df)

        df['year']   = df['time'].dt.year
        df['season'] = df['month'].apply(get_season)

        FEAT_COLS[:] = [c for c in ['temperature', 'humidity', 'pressure', 'rain',
                                     'cloud_cover', 'wind_speed',
                                     'sin_h', 'cos_h', 'sin_m', 'cos_m']
                        if c in df.columns]

        historical_data = df
        print(f"✅ Historical data: {len(df):,} rows | {df['time'].min().date()} → {df['time'].max().date()}")
        return True
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def _formula_aqi(df: pd.DataFrame) -> pd.Series:
    t  = pd.to_numeric(df.get('temperature', 24), errors='coerce').fillna(24)
    h  = pd.to_numeric(df.get('humidity', 60),    errors='coerce').fillna(60)
    w  = pd.to_numeric(df.get('wind_speed', 10),  errors='coerce').fillna(10)
    cl = pd.to_numeric(df.get('cloud_cover', 35), errors='coerce').fillna(35)
    r  = pd.to_numeric(df.get('rain', 0),         errors='coerce').fillna(0)
    m  = df['time'].dt.month if 'time' in df.columns else pd.Series([6]*len(df))
    seasonal = np.where(m.isin([10, 11, 12, 1, 2]), 35,
               np.where(m.isin([3, 4, 5]), 15, 5))
    aqi = 40 + 1.7*t + 0.35*h + 0.08*cl - 1.1*w + seasonal - np.minimum(r*25, 35)
    return np.clip(np.round(aqi), 20, 450).astype(int)

def load_predictions_data() -> bool:
    global predictions_data
    if Path('temperature_predictions.csv').exists():
        predictions_data = pd.read_csv('temperature_predictions.csv')
        print(f"✅ Prediction CSV: {len(predictions_data)} rows")
        return True
    return False


def _run_model(model, X_np: np.ndarray) -> np.ndarray:
    """Run a WeatherLSTM on a (1, SEQ_LEN, F) array, returns (SEQ_LEN,)."""
    X_t = torch.tensor(X_np, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        if torch.cuda.is_available():
            with autocast():
                out = model(X_t)
        else:
            out = model(X_t)
    return out.squeeze(0).squeeze(-1).cpu().numpy()

def _get_latest_sequence() -> np.ndarray | None:
    if historical_data is None or feat_scaler is None or len(FEAT_COLS) == 0:
        return None
    tail = historical_data.tail(SEQ_LEN)
    if len(tail) < SEQ_LEN:
        return None
    try:
        scaled = feat_scaler.transform(tail[FEAT_COLS])
        return scaled[np.newaxis]          # (1, 168, F)
    except Exception:
        return None


def generate_forecast(days: int = 7) -> list | None:
    if historical_data is None:
        return None

    now   = datetime.now()
    hours = days * 24

    
    ml_temps = None
    ml_aqis  = None
    seq = _get_latest_sequence()

    if seq is not None and temp_model is not None and temp_scaler is not None:
        try:
            raw = _run_model(temp_model, seq)[:hours]  
            ml_temps = temp_scaler.inverse_transform(raw.reshape(-1,1)).flatten()
        except Exception as e:
            print(f"⚠️  Temp model inference failed: {e}")

    if seq is not None and aqi_model is not None and aqi_scaler is not None:
        try:
            raw = _run_model(aqi_model, seq)[:hours]
            ml_aqis = aqi_scaler.inverse_transform(raw.reshape(-1,1)).flatten()
            ml_aqis = np.clip(ml_aqis, 20, 500)
        except Exception as e:
            print(f"⚠️  AQI model inference failed: {e}")

    
    hp = historical_data.groupby(['month', 'hour']).agg(
        temp_mean  = ('temperature', 'mean'),
        temp_std   = ('temperature', 'std'),
        rain_prob  = ('rain', lambda s: float((s > 0).mean()) * 100),
        aqi_mean   = ('AQI', 'mean'),
        aqi_std    = ('AQI', 'std'),
    )
    gm = historical_data.groupby('month').agg(
        temp_mean  = ('temperature', 'mean'),
        temp_std   = ('temperature', 'std'),
        rain_prob  = ('rain', lambda s: float((s > 0).mean()) * 100),
        aqi_mean   = ('AQI', 'mean'),
        aqi_std    = ('AQI', 'std'),
    )
    gd = {
        'temp_mean': float(historical_data['temperature'].mean()),
        'temp_std':  float(historical_data['temperature'].std()),
        'rain_prob': float((historical_data['rain'] > 0).mean()) * 100,
        'aqi_mean':  float(historical_data['AQI'].mean()),
        'aqi_std':   float(historical_data['AQI'].std()),
    }

    
    if TORCH_OK and torch.cuda.is_available():
        idx = torch.arange(hours, device=DEVICE, dtype=torch.float32)
        t_off  = (1.4 * torch.sin(idx / 1.8)).cpu().numpy()
        a_off  = (4.8 * torch.cos(idx / 2.2)).cpu().numpy()
    else:
        idx   = np.arange(hours, dtype=float)
        t_off = 1.4 * np.sin(idx / 1.8)
        a_off = 4.8 * np.cos(idx / 2.2)

    forecast = []
    for day in range(days):
        dt  = now + timedelta(days=day)
        key = (dt.month, 14)
        p   = hp.loc[key] if key in hp.index else (gm.loc[dt.month] if dt.month in gm.index else pd.Series(gd))

        t_mn  = float(p.get('temp_mean', gd['temp_mean']))
        t_sd  = float(p.get('temp_std',  gd['temp_std']) or gd['temp_std'])
        r_prob= float(np.clip(p.get('rain_prob', gd['rain_prob']), 5, 95))
        a_mn  = float(p.get('aqi_mean',  gd['aqi_mean']))
        a_sd  = float(p.get('aqi_std',   gd['aqi_std']) or gd['aqi_std'])

        
        if ml_temps is not None and day * 24 < len(ml_temps):
            temperature = float(np.clip(ml_temps[day * 24 : (day+1)*24].mean(), 2, 50))
        else:
            temperature = float(np.clip(t_mn + t_off[day*24] + t_sd*0.05, 2, 50))

        if ml_aqis is not None and day * 24 < len(ml_aqis):
            aqi = int(np.clip(round(ml_aqis[day * 24 : (day+1)*24].mean()), 20, 500))
        else:
            aqi = int(np.clip(round(a_mn + a_off[day*24]), 20, 500))

        forecast.append({
            'date':          dt.strftime('%Y-%m-%d'),
            'temperature':   round(temperature, 1),
            'rainfall':      'Yes' if r_prob >= 45 else 'No',
            'rain_probability': round(r_prob, 1),
            'aqi':           aqi,
            'aqi_category':  classify_aqi(aqi),
            'ml_powered':    temp_model is not None,
        })
    return forecast

def generate_hourly_for_chart(days: int = 7) -> dict | None:
    
    if historical_data is None:
        return None

    hours = days * 24
    now   = datetime.now()
    seq   = _get_latest_sequence()

   
    if seq is not None and temp_model is not None and temp_scaler is not None:
        try:
            raw    = _run_model(temp_model, seq)[:hours]
            temps  = temp_scaler.inverse_transform(raw.reshape(-1,1)).flatten().tolist()
        except Exception:
            temps = None
    else:
        temps = None

    if temps is None:
        hp = historical_data.groupby(['month', 'hour'])['temperature'].mean()
        temps = []
        for h in range(hours):
            dt  = now + timedelta(hours=h)
            key = (dt.month, dt.hour)
            val = float(hp.get(key, historical_data['temperature'].mean()))
            temps.append(round(val + 1.4*np.sin(h/1.8), 2))


    if seq is not None and aqi_model is not None and aqi_scaler is not None:
        try:
            raw  = _run_model(aqi_model, seq)[:hours]
            aqis = np.clip(aqi_scaler.inverse_transform(raw.reshape(-1,1)).flatten(), 20, 500).tolist()
        except Exception:
            aqis = None
    else:
        aqis = None

    if aqis is None:
        ha = historical_data.groupby(['month', 'hour'])['AQI'].mean()
        aqis = []
        for h in range(hours):
            dt  = now + timedelta(hours=h)
            key = (dt.month, dt.hour)
            val = float(ha.get(key, historical_data['AQI'].mean()))
            aqis.append(int(np.clip(round(val + 4.8*np.cos(h/2.2)), 20, 500)))


    hum_profile  = historical_data.groupby(['month','hour'])['humidity'].mean()   if 'humidity'   in historical_data.columns else None
    wind_profile = historical_data.groupby(['month','hour'])['wind_speed'].mean() if 'wind_speed'  in historical_data.columns else None

    humidities  = []
    wind_speeds = []
    for h in range(hours):
        dt  = now + timedelta(hours=h)
        key = (dt.month, dt.hour)
        if hum_profile is not None:
            hv = float(hum_profile.get(key, historical_data['humidity'].mean()))
        else:
            hv = 60.0
        humidities.append(round(hv + 3 * np.sin(h / 3.5), 1))

        if wind_profile is not None:
            wv = float(wind_profile.get(key, historical_data['wind_speed'].mean()))
        else:
            wv = 10.0
        wind_speeds.append(round(max(0, wv + 2 * np.cos(h / 4.0)), 1))

    labels = [(now + timedelta(hours=h)).strftime('%b %d %H:%M') for h in range(hours)]
    return {'labels': labels, 'temperatures': temps, 'aqis': aqis,
            'humidities': humidities, 'wind_speeds': wind_speeds,
            'ml_powered': temp_model is not None}


def _fig_to_b64() -> str:
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return b64

def create_yearly_analysis() -> str | None:
    if historical_data is None:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle('Yearly Weather & AQI Analysis — Noida', fontsize=18, fontweight='bold')

    mp = historical_data.groupby(['year', 'month'])['temperature'].mean().reset_index()
    pv = mp.pivot(index='month', columns='year', values='temperature')
    sns.heatmap(pv, ax=axes[0,0], cmap='RdYlBu_r', center=20,
                annot=True, fmt='.1f', cbar_kws={'label': '°C'})
    axes[0,0].set_title('Monthly Temperature Heatmap')

    dp = historical_data.groupby(['month','hour'])['temperature'].mean().reset_index()
    dv = dp.pivot(index='hour', columns='month', values='temperature')
    sns.heatmap(dv, ax=axes[0,1], cmap='RdYlBu_r', center=20, cbar_kws={'label': '°C'})
    axes[0,1].set_title('Hour × Month Temperature Pattern')

    sns.boxplot(data=historical_data, x='season', y='temperature',
                order=['Winter','Spring','Summer','Autumn'], ax=axes[1,0])
    axes[1,0].set_title('Seasonal Temperature Distribution')

    yt = historical_data.groupby('year')['temperature'].agg(['mean','min','max']).reset_index()
    axes[1,1].plot(yt['year'], yt['mean'], marker='o', lw=2, label='Avg Temp')
    axes[1,1].fill_between(yt['year'], yt['min'], yt['max'], alpha=0.25, label='Range')
    if 'AQI' in historical_data.columns:
        ya = historical_data.groupby('year')['AQI'].mean().reset_index()
        ax2 = axes[1,1].twinx()
        ax2.plot(ya['year'], ya['AQI'], marker='s', lw=2, color='purple', label='Avg AQI')
        ax2.set_ylabel('AQI', color='purple')
        ax2.legend(loc='upper right')
    axes[1,1].set_title('Yearly Trends')
    axes[1,1].legend(loc='upper left')
    axes[1,1].grid(alpha=0.3)

    plt.tight_layout()
    return _fig_to_b64()

def create_30day_comparison() -> str | None:
    if historical_data is None:
        return None
    end   = historical_data['time'].max()
    start = end - timedelta(days=30)
    rec   = historical_data[(historical_data['time'] >= start)].copy()
    if rec.empty:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('30-Day Analysis vs Predictions', fontsize=15, fontweight='bold')

    rd = rec.groupby(rec['time'].dt.date)['temperature'].agg(['mean','min','max']).reset_index()
    rd['time'] = pd.to_datetime(rd['time'])
    axes[0,0].plot(rd['time'], rd['mean'], 'b-o', lw=2, label='Actual avg')
    axes[0,0].fill_between(rd['time'], rd['min'], rd['max'], alpha=0.25, color='blue', label='Range')
    axes[0,0].set_title('Recent 30 Days Temperature')
    axes[0,0].legend(); axes[0,0].grid(alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=45)

    month = rec['time'].dt.month.iloc[0]
    hist_m = historical_data[historical_data['month'] == month]
    ha = hist_m.groupby('hour')['temperature'].mean()
    pred_temps = None
    if predictions_data is not None and 'predicted_temperature' in predictions_data.columns:
        pred_temps = predictions_data['predicted_temperature'].values[:24]
    else:
        gen = generate_forecast(1)
        if gen:
            pred_temps = [gen[0]['temperature']]*24

    if pred_temps is not None:
        axes[0,1].plot(range(24), pred_temps[:24], 'r-o', lw=2, label='Predicted')
    axes[0,1].plot(ha.index, ha.values, 'g-s', lw=2, label='Hist avg')
    axes[0,1].set_title('24-h Prediction vs Historical')
    axes[0,1].legend(); axes[0,1].grid(alpha=0.3)
    axes[0,1].set_xlabel('Hour')

    axes[1,0].hist(rec['temperature'], bins=30, alpha=0.6, color='blue', label='Recent', density=True)
    if pred_temps is not None:
        axes[1,0].hist(np.tile(pred_temps, 7), bins=30, alpha=0.6, color='red', label='Pred', density=True)
    axes[1,0].set_title('Temperature Distribution')
    axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

    if 'AQI' in rec.columns:
        ra = rec.groupby(rec['time'].dt.date)['AQI'].mean().reset_index()
        ra['time'] = pd.to_datetime(ra['time'])
        axes[1,1].plot(ra['time'], ra['AQI'], 'purple', lw=2, marker='o', label='AQI')
        axes[1,1].axhline(100, ls='--', color='orange', label='Moderate threshold')
        axes[1,1].axhline(150, ls='--', color='red',    label='Unhealthy threshold')
        axes[1,1].set_title('30-Day AQI Trend')
        axes[1,1].legend(); axes[1,1].grid(alpha=0.3)
        axes[1,1].tick_params(axis='x', rotation=45)
    else:
        axes[1,1].axis('off')

    plt.tight_layout()
    return _fig_to_b64()

def get_improved_predictions() -> list | None:
    gen = generate_forecast(7)
    if not gen:
        return None
    out = []
    for day in gen:
        for h in range(24):
            diurnal = 2.8 * np.sin(((h - 6) / 24) * 2 * np.pi)
            out.append(float(np.clip(day['temperature'] + diurnal, 2, 50)))
    return out[:168]


print("\n── Loading data ──────────────────────────────────────────")
historical_loaded  = load_historical_data()
predictions_loaded = load_predictions_data()
print("── Loading models ────────────────────────────────────────")
load_models()
print("─────────────────────────────────────────────────────────\n")


@app.route('/')
def home():
    return render_template('modern_weather.html')

@app.route('/api/noida-weather')
def noida_weather():
    try:
        fc = generate_forecast(7)
        if not fc:
            return jsonify({'error': 'Data not ready'}), 500
        return jsonify({
            'status': 'success',
            'location': NOIDA['name'],
            'forecast': fc,
            'ml_powered': temp_model is not None,
            'compute_backend': get_compute_label(),
            'generated_at': datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/hourly-forecast')
def hourly_forecast():
    
    try:
        data = generate_hourly_for_chart(7)
        if not data:
            return jsonify({'error': 'Data not ready'}), 500
        return jsonify({'status': 'success', **data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/yearly-analysis')
def yearly_analysis():
    try:
        img = create_yearly_analysis()
        if img is None:
            return jsonify({'error': 'No historical data'}), 500
        return jsonify({'status': 'success', 'image': f'data:image/png;base64,{img}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/30-day-comparison')
def thirty_day():
    try:
        img = create_30day_comparison()
        if img is None:
            return jsonify({'error': 'No historical data'}), 500
        return jsonify({'status': 'success', 'image': f'data:image/png;base64,{img}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/improved-predictions')
def improved_predictions():
    try:
        preds = get_improved_predictions()
        if preds is None:
            return jsonify({'error': 'No data'}), 500
        # Also get hourly AQI
        hourly = generate_hourly_for_chart(7)
        aqis   = hourly['aqis'] if hourly else [int(historical_data['AQI'].mean())]*168
        return jsonify({
            'status': 'success',
            'predictions': {
                'hours':        list(range(168)),
                'temperatures': preds,
                'aqis':         aqis,
                'min_temp':     float(min(preds)),
                'max_temp':     float(max(preds)),
                'mean_temp':    float(np.mean(preds)),
                'generated_at': datetime.now().isoformat(),
            },
            'ml_powered': temp_model is not None,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/historical-stats')
def historical_stats():
    try:
        if historical_data is None:
            return jsonify({'error': 'No data'}), 500
        s = {
            'total_records': len(historical_data),
            'date_range': {
                'start': historical_data['time'].min().isoformat(),
                'end':   historical_data['time'].max().isoformat(),
            },
            'temperature': {
                'mean': round(float(historical_data['temperature'].mean()), 2),
                'min':  round(float(historical_data['temperature'].min()), 2),
                'max':  round(float(historical_data['temperature'].max()), 2),
            },
            'aqi': {
                'mean': round(float(historical_data['AQI'].mean()), 1),
                'min':  int(historical_data['AQI'].min()),
                'max':  int(historical_data['AQI'].max()),
            },
            'monthly_avg_temp': historical_data.groupby('month')['temperature'].mean().round(2).to_dict(),
            'monthly_avg_aqi':  historical_data.groupby('month')['AQI'].mean().round(1).to_dict(),
        }
        return jsonify({'status': 'success', 'statistics': s})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'historical_loaded': historical_loaded,
        'predictions_loaded': predictions_loaded,
        'temp_model_loaded': temp_model is not None,
        'aqi_model_loaded':  aqi_model is not None,
        'compute_backend': get_compute_label(),
        'timestamp': datetime.now().isoformat(),
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
