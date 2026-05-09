

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


plt.style.use('seaborn-v0_8')
sns.set_palette("husl")



print("🔍 Debug CUDA environment")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.version.cuda else 'N/A'}")
print(f"Device count: {torch.cuda.device_count()}")
print()


if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"✅ Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️ CUDA not available in this environment, using CPU")
    print("This might be due to:")
    print("  1. Jupyter kernel not having GPU access")
    print("  2. Different conda/virtual environment")
    print("  3. GPU driver issues")
    device = torch.device('cpu')
    print(f"Using device: {device}")



print("\n📂 Data Loading & Preprocessing")
print("=" * 50)


file_name = "Filtered_Weather_Forecasting_Datasetls.xlsx"
if not os.path.exists(file_name):
    raise FileNotFoundError(f"{file_name} not found in the current directory.")

df = pd.read_excel(file_name)
df['time'] = pd.to_datetime(df['time'])


if 'temperature' not in df.columns:
    df.rename(columns={'temperature_2m (°C)': 'temperature'}, inplace=True)


feature_columns = ['temperature', 'relative_humidity', 'pressure_msl', 'rain']


df['hour_of_day'] = df['time'].dt.hour
df['day_of_week'] = df['time'].dt.dayofweek


feature_columns.extend(['hour_of_day', 'day_of_week'])


available_features = [col for col in feature_columns if col in df.columns]
if len(available_features) < len(feature_columns):
    missing_features = set(feature_columns) - set(available_features)
    print(f"⚠️ Warning: Missing features: {missing_features}")
    print(f"Available features: {available_features}")
    feature_columns = available_features

df = df[['time'] + feature_columns].dropna()
print(f"Dataset shape: {df.shape}")
print(f"Features: {feature_columns}")
print(f"Temperature range: {df['temperature'].min():.2f}°C to {df['temperature'].max():.2f}°C")


print("\nFeature Statistics:")
for feature in feature_columns:
    if feature in df.columns:
        print(f"• {feature}: {df[feature].min():.2f} to {df[feature].max():.2f}")


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()


plot_features = ['temperature', 'relative_humidity', 'pressure_msl', 'rain']
for i, feature in enumerate(plot_features):
    if feature in df.columns and i < 4:
        axes[i].plot(df['time'], df[feature], alpha=0.7, color=plt.cm.tab10(i))
        axes[i].set_title(f'Raw {feature.replace("_", " ").title()} Data')
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel(feature.replace("_", " ").title())
        axes[i].grid(True)
    else:
        axes[i].axis('off')

plt.tight_layout()
plt.show()


if 'hour_of_day' in df.columns and 'day_of_week' in df.columns:
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['hour_of_day'], bins=24, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Hour of Day Distribution')
    plt.xlabel('Hour')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.hist(df['day_of_week'], bins=7, alpha=0.7, color='green', edgecolor='black')
    plt.title('Day of Week Distribution')
    plt.xlabel('Day (0=Mon, 6=Sun)')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()



print("\n🔢 Data Preprocessing and Sequence Creation")
print("=" * 50)


scaler = StandardScaler()
data = scaler.fit_transform(df[feature_columns])
print(f"Data shape after scaling: {data.shape}")
print(f"Number of features: {len(feature_columns)}")


def create_dataset(data, input_len=168, output_len=168, stride=1, target_feature_idx=0):
    """
    Create input-output sequences for LSTM training using sliding window
    
    Args:
        data: Standardized time series data (samples, features)
        input_len: Length of input sequence (default: 168 hours = 7 days)
        output_len: Length of output sequence (default: 168 hours = 7 days)
        stride: Step size for sliding window (default: 1 for maximum samples)
        target_feature_idx: Index of target feature (temperature) for output
    
    Returns:
        X: Input sequences of shape (num_samples, input_len, num_features)
        y: Output sequences of shape (num_samples, output_len, 1) - only temperature
    """
    X, y = [], []
    for i in range(0, len(data) - input_len - output_len + 1, stride):
        # Input: all features for input_len timesteps
        X.append(data[i:i+input_len, :])  # shape: (input_len, num_features)
        # Output: only temperature for output_len timesteps
        y.append(data[i+input_len:i+input_len+output_len, target_feature_idx:target_feature_idx+1])  # shape: (output_len, 1)
    return np.array(X), np.array(y)


temp_idx = feature_columns.index('temperature')

X, y = create_dataset(data, input_len=168, output_len=168, stride=1, target_feature_idx=temp_idx)
print(f"Input sequences shape: {X.shape}")
print(f"Output sequences shape: {y.shape}")
print(f"Target feature index (temperature): {temp_idx}")



print("\n🧭 Dataset Preparation")
print("=" * 50)


X = torch.tensor(X, dtype=torch.float32)  
y = torch.tensor(y, dtype=torch.float32)  

print(f"X tensor shape: {X.shape}")
print(f"y tensor shape: {y.shape}")
print(f"Number of input features: {X.shape[2]}")


split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")


train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)



class CombinedLoss(nn.Module):
    
    def __init__(self, mae_weight=0.3, rmse_weight=0.4, smooth_l1_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.mae_weight = mae_weight
        self.rmse_weight = rmse_weight
        self.smooth_l1_weight = smooth_l1_weight
        
        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
    
    def forward(self, predictions, targets):
        mae = self.mae_loss(predictions, targets)
        rmse = torch.sqrt(self.mse_loss(predictions, targets))
        smooth_l1 = self.smooth_l1_loss(predictions, targets)
        
        combined_loss = (
            self.mae_weight * mae +
            self.rmse_weight * rmse +
            self.smooth_l1_weight * smooth_l1
        )
        
        return combined_loss, {'mae': mae.item(), 'rmse': rmse.item(), 'smooth_l1': smooth_l1.item()}



def create_curriculum_dataset(data, input_len=168, output_len=168, stride=1, target_feature_idx=0):
    
    X, y = [], []
    for i in range(0, len(data) - input_len - output_len + 1, stride):
        X.append(data[i:i+input_len, :])  # Input: all features
        y.append(data[i+input_len:i+input_len+output_len, target_feature_idx:target_feature_idx+1])  # Output: temperature only
    return np.array(X), np.array(y)



print("\n🧠 Enhanced LSTM Model with Regularization")
print("=" * 50)

class CompactForecastModel(nn.Module):
    def __init__(self, input_size, use_gru=False, use_cnn=False, dropout_rate=0.2):
        super(CompactForecastModel, self).__init__()
        self.use_cnn = use_cnn
        self.use_gru = use_gru
        self.input_size = input_size

        if use_cnn:
            self.cnn = nn.Sequential(
                nn.Conv1d(input_size, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU()
            )
            rnn_input_size = 64
        else:
            rnn_input_size = input_size

        if use_gru:
            self.rnn = nn.GRU(
                input_size=rnn_input_size,
                hidden_size=64,
                num_layers=2,
                batch_first=True,
                dropout=dropout_rate,
            )
        else:
            self.rnn = nn.LSTM(
                input_size=rnn_input_size,
                hidden_size=64,
                num_layers=2,
                batch_first=True,
                dropout=dropout_rate,
            )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        if self.use_cnn:
            x = x.permute(0, 2, 1)  
            x = self.cnn(x)
            x = x.permute(0, 2, 1)  
        
        rnn_out, _ = self.rnn(x)
        rnn_out = self.dropout(rnn_out)
        out = self.fc(rnn_out)
        return out

class ImprovedForecastNet(nn.Module):
    
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, 1)  

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)  


num_features = len(feature_columns)
model = CompactForecastModel(input_size=num_features, use_gru=False, use_cnn=True).to(device)
criterion = CombinedLoss(mae_weight=0.3, rmse_weight=0.4, smooth_l1_weight=0.3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=5
)

print(f"Model input size: {num_features} features")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Using combined loss: MAE (30%) + RMSE (40%) + SmoothL1 (30%)")


curriculum_stages = [(24, 1), (48, 2), (168, 7)]
for output_len, increment in curriculum_stages:
    print(f"\nCurriculum Stage: {output_len} hour forecast (Increment: {increment} days)")
    X_curriculum, y_curriculum = create_curriculum_dataset(data, input_len=168, output_len=output_len, stride=168)

    X_curriculum = torch.tensor(X_curriculum, dtype=torch.float32)
    y_curriculum = torch.tensor(y_curriculum, dtype=torch.float32)

    split = int(0.8 * len(X_curriculum))
    X_train_curriculum, X_val_curriculum = X_curriculum[:split], X_curriculum[split:]
    y_train_curriculum, y_val_curriculum = y_curriculum[:split], y_curriculum[split:]

    train_loader_curriculum = DataLoader(TensorDataset(X_train_curriculum, y_train_curriculum), batch_size=64, shuffle=True)
    val_loader_curriculum = DataLoader(TensorDataset(X_val_curriculum, y_val_curriculum), batch_size=64)

    
    for epoch in range(1, 21):
        model.train()
        train_loss = 0
        for xb, yb in train_loader_curriculum:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            
            pred = pred[:, :output_len, :]
            loss, _ = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader_curriculum:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                
                pred = pred[:, :output_len, :]
                val_loss += criterion(pred, yb)[0].item()

        avg_train_loss = train_loss / len(train_loader_curriculum)
        avg_val_loss = val_loss / len(val_loader_curriculum)
        scheduler.step(avg_val_loss)

        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d} \u2212 train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}")



print("Training complete! Curriculum learning stages fully executed.")




print("\n🚀 Enhanced Training Loop with Metrics Tracking")
print("=" * 50)


train_losses = []
val_losses = []
learning_rates = []


best_val_loss = float('inf')
patience = 20
counter = 0

print("🚀 Starting enhanced training...")
print()


for epoch in range(1, 151):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

       
        
        pred = model(xb)
        loss, _ = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_loss += criterion(pred, yb)[0].item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    learning_rates.append(optimizer.param_groups[0]['lr'])
    
    
    scheduler.step(avg_val_loss)
    
    if epoch % 10 == 0 or epoch < 10:
        print(f"Epoch {epoch:3d}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}, lr={optimizer.param_groups[0]['lr']:.2e}")

    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        best_model = model.state_dict().copy()
    else:
        counter += 1
        if counter >= patience:
            print(f"\n🛑 Early stopping triggered at epoch {epoch}")
            break

print(f"\n✅ Training completed! Best validation loss: {best_val_loss:.6f}")




model_path = "best_model.pth"
torch.save(best_model, model_path)
print(f"Model saved to {model_path}")


import joblib
scaler_path = "scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")



print("\n📊 Training Progress Visualization")
print("=" * 50)


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))


ax1.plot(train_losses, label='Training Loss', color='blue', alpha=0.7)
ax1.plot(val_losses, label='Validation Loss', color='red', alpha=0.7)
ax1.set_title('Training Progress')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)


ax2.plot(learning_rates, color='green', alpha=0.7)
ax2.set_title('Learning Rate Schedule')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Learning Rate')
ax2.set_yscale('log')
ax2.grid(True)


ax3.hist(train_losses, bins=30, alpha=0.7, label='Training Loss', color='blue')
ax3.hist(val_losses, bins=30, alpha=0.7, label='Validation Loss', color='red')
ax3.set_title('Loss Distribution')
ax3.set_xlabel('Loss Value')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.grid(True)


ax4.plot(np.convolve(train_losses, np.ones(5)/5, mode='valid'), label='Training Loss (5-epoch MA)', color='blue')
ax4.plot(np.convolve(val_losses, np.ones(5)/5, mode='valid'), label='Validation Loss (5-epoch MA)', color='red')
ax4.set_title('Loss Convergence (Moving Average)')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Loss')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.show()



print("\n📈 Model Evaluation")
print("=" * 50)


model.load_state_dict(best_model)
model.eval()


predictions = []
actuals = []

with torch.no_grad():
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        predictions.append(pred.cpu().numpy())
        actuals.append(yb.cpu().numpy())


predictions = np.concatenate(predictions, axis=0)
actuals = np.concatenate(actuals, axis=0)


temp_scaler = StandardScaler()
temp_scaler.fit(df[['temperature']])

predictions_inv = temp_scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(predictions.shape)
actuals_inv = temp_scaler.inverse_transform(actuals.reshape(-1, 1)).reshape(actuals.shape)


mae = mean_absolute_error(actuals_inv.flatten(), predictions_inv.flatten())
mse = mean_squared_error(actuals_inv.flatten(), predictions_inv.flatten())
rmse = np.sqrt(mse)
r2 = r2_score(actuals_inv.flatten(), predictions_inv.flatten())

print(f"📊 Model Performance Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.4f}°C")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}°C")
print(f"R² Score: {r2:.4f}")



print("\n🎨 Visualization of Predictions")
print("=" * 50)


sample_idx = 0
sample_pred = predictions_inv[sample_idx]
sample_actual = actuals_inv[sample_idx]

plt.figure(figsize=(15, 6))
time_steps = range(len(sample_pred))
plt.plot(time_steps, sample_actual, label='Actual', color='blue', linewidth=2)
plt.plot(time_steps, sample_pred, label='Predicted', color='red', linewidth=2, linestyle='--')
plt.title('Temperature Forecast vs Actual (Sample Prediction)')
plt.xlabel('Hours')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True)
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i in range(4):
    sample_pred = predictions_inv[i]
    sample_actual = actuals_inv[i]
    time_steps = range(len(sample_pred))
    
    axes[i].plot(time_steps, sample_actual, label='Actual', color='blue', linewidth=2)
    axes[i].plot(time_steps, sample_pred, label='Predicted', color='red', linewidth=2, linestyle='--')
    axes[i].set_title(f'Sample {i+1}')
    axes[i].set_xlabel('Hours')
    axes[i].set_ylabel('Temperature (°C)')
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.show()



print("\n🔍 Feature Importance and Error Analysis")
print("=" * 50)


errors = predictions_inv - actuals_inv
abs_errors = np.abs(errors)


print(f"Error Statistics:")
print(f"Mean Error: {np.mean(errors):.4f}°C")
print(f"Std Error: {np.std(errors):.4f}°C")
print(f"Max Absolute Error: {np.max(abs_errors):.4f}°C")
print(f"95th Percentile Error: {np.percentile(abs_errors, 95):.4f}°C")


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(errors.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.title('Error Distribution')
plt.xlabel('Error (°C)')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.hist(abs_errors.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
plt.title('Absolute Error Distribution')
plt.xlabel('Absolute Error (°C)')
plt.ylabel('Frequency')
plt.grid(True)

plt.tight_layout()
plt.show()


print("\n📝 Summary and Conclusion")
print("=" * 50)
print(f"🎯 Model Summary:")
print(f"• Architecture: Enhanced Multi-variate LSTM with CNN preprocessing")
print(f"• Input sequence length: 168 hours (7 days)")
print(f"• Output sequence length: 168 hours (7 days)")
print(f"• Input features: {len(feature_columns)} ({', '.join(feature_columns)})")
print(f"• Hidden size: 64 units")
print(f"• Dropout rate: 0.2")
print(f"• Training epochs: {len(train_losses)}")
print(f"• Best validation loss: {best_val_loss:.6f}")
print(f"• Best validation loss: {best_val_loss:.6f}")

print(f"\n📊 Performance Metrics:")
print(f"• MAE: {mae:.4f}°C")
print(f"• RMSE: {rmse:.4f}°C")
print(f"• R² Score: {r2:.4f}")

print(f"\n🚀 Key Features:")
print(f"• ✅ Multi-variate input with temporal features")
print(f"• ✅ CNN preprocessing for feature extraction")
print(f"• ✅ Early stopping to prevent overfitting")
print(f"• ✅ Learning rate scheduling")
print(f"• ✅ Gradient clipping for stable training")
print(f"• ✅ Dropout regularization")
print(f"• ✅ GPU acceleration")

print(f"\n💡 Recommendations:")
if mae < 2.0:
    print(f"• 🟢 Excellent performance! MAE < 2.0°C")
elif mae < 3.0:
    print(f"• 🟡 Good performance! MAE < 3.0°C")
else:
    print(f"• 🔴 Consider model improvements")

print(f"\n✨ Training complete! Model ready for deployment.")
