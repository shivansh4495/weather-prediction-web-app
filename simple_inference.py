#!/usr/bin/env python3
"""
Simple Weather Forecasting Inference Script
==========================================

This script provides a minimal interface for weather forecasting inference
using a pre-trained model and scaler (pkl file).

Usage:
    python simple_inference.py
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Model Architecture (Must match training exactly)
# =============================================================================

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
            x = x.permute(0, 2, 1)  # (batch, features, seq_len)
            x = self.cnn(x)
            x = x.permute(0, 2, 1)  # (batch, seq_len, features)
        
        rnn_out, _ = self.rnn(x)
        rnn_out = self.dropout(rnn_out)
        out = self.fc(rnn_out)
        return out

# =============================================================================
# Simple Inference Function
# =============================================================================

def load_model_and_scaler():
    """Load the trained model and scaler"""
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load scaler
    scaler = joblib.load('scaler.pkl')
    print("✅ Scaler loaded successfully")
    
    # Load model - check actual features from scaler
    # Based on scaler inspection, the model was trained with 3 features
    # The scaler means suggest: [temperature, hour_of_day, day_of_week]
    feature_columns = ['temperature', 'hour_of_day', 'day_of_week']
    model = CompactForecastModel(
        input_size=len(feature_columns),
        use_gru=False,
        use_cnn=True,
        dropout_rate=0.2
    ).to(device)
    
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()
    print("✅ Model loaded successfully")
    
    return model, scaler, device

def prepare_input_data(data, scaler):
    """Prepare input data for inference"""
    
    # Feature columns expected by the model (only 3 features)
    feature_columns = ['temperature', 'hour_of_day', 'day_of_week']
    
    # If data is a DataFrame, add temporal features
    if isinstance(data, pd.DataFrame):
        if 'time' in data.columns:
            data['time'] = pd.to_datetime(data['time'])
            data['hour_of_day'] = data['time'].dt.hour
            data['day_of_week'] = data['time'].dt.dayofweek
        
        # Ensure we have all required features
        if 'hour_of_day' not in data.columns:
            data['hour_of_day'] = 12  # Default to noon
        if 'day_of_week' not in data.columns:
            data['day_of_week'] = 0   # Default to Monday
        
        # Select feature columns
        data = data[feature_columns].dropna()
    
    # Scale the data
    scaled_data = scaler.transform(data)
    
    return scaled_data

def create_sequences(data, input_length=168):
    """Create input sequences for the model"""
    
    sequences = []
    for i in range(len(data) - input_length + 1):
        sequences.append(data[i:i + input_length])
    
    return np.array(sequences)

def predict_temperature(model, scaler, input_data, device):
    """Make temperature predictions"""
    
    # Prepare input data
    scaled_data = prepare_input_data(input_data, scaler)
    
    # Check if we have enough data
    if len(scaled_data) < 168:
        raise ValueError(f"Need at least 168 hours of data, got {len(scaled_data)}")
    
    # Create sequences
    sequences = create_sequences(scaled_data)
    
    if len(sequences) == 0:
        raise ValueError("No valid sequences created")
    
    # Convert to tensor
    X = torch.tensor(sequences, dtype=torch.float32).to(device)
    
    # Make predictions
    predictions = []
    with torch.no_grad():
        for i in range(0, len(X), 32):  # Process in batches of 32
            batch = X[i:i + 32]
            pred = model(batch)
            predictions.append(pred.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    
    return predictions

def inverse_transform_temperature(predictions, original_temp_data):
    """Convert scaled predictions back to actual temperature values"""
    
    # Create a simple scaler for temperature only
    from sklearn.preprocessing import StandardScaler
    temp_scaler = StandardScaler()
    temp_scaler.fit(original_temp_data.reshape(-1, 1))
    
    # Inverse transform
    predictions_inv = temp_scaler.inverse_transform(
        predictions.reshape(-1, 1)
    ).reshape(predictions.shape)
    
    return predictions_inv

# =============================================================================
# Main Function
# =============================================================================

def main():
    """Main inference function"""
    
    print("🌤️ Weather Forecasting - Simple Inference")
    print("=" * 50)
    
    # Load model and scaler
    model, scaler, device = load_model_and_scaler()
    
    # Example: Create sample data for demonstration
    # Replace this with your actual data loading
    print("\n📊 Creating sample data for demonstration...")
    
    # Sample data (replace with your actual data)
    sample_data = pd.DataFrame({
        'temperature': np.random.normal(20, 5, 200),  # 200 hours of temperature data
        'relative_humidity': np.random.normal(60, 10, 200),
        'pressure_msl': np.random.normal(1013, 5, 200),
        'rain': np.random.exponential(0.5, 200),
        'hour_of_day': np.tile(np.arange(24), 200//24 + 1)[:200],
        'day_of_week': np.tile(np.arange(7), 200//7 + 1)[:200]
    })
    
    print(f"Sample data shape: {sample_data.shape}")
    
    # Make predictions
    print("\n🔮 Making predictions...")
    predictions = predict_temperature(model, scaler, sample_data, device)
    
    print(f"Generated {len(predictions)} prediction sequences")
    print(f"Each sequence predicts {predictions.shape[1]} hours ahead")
    
    # Convert back to actual temperature values
    predictions_celsius = inverse_transform_temperature(
        predictions, 
        sample_data['temperature'].values
    )
    
    print(f"\n📈 Prediction Results:")
    print(f"First prediction sequence (next 168 hours):")
    print(f"Min temperature: {predictions_celsius[0].min():.2f}°C")
    print(f"Max temperature: {predictions_celsius[0].max():.2f}°C")
    print(f"Mean temperature: {predictions_celsius[0].mean():.2f}°C")
    
    # Save predictions to CSV
    output_df = pd.DataFrame({
        'hour': range(168),
        'predicted_temperature': predictions_celsius[0].flatten()
    })
    
    output_df.to_csv('temperature_predictions.csv', index=False)
    print(f"\n💾 Predictions saved to 'temperature_predictions.csv'")
    
    return predictions_celsius

# =============================================================================
# Custom Data Loading Function
# =============================================================================

def load_your_data(file_path):
    """
    Load your own data for inference
    
    Args:
        file_path: Path to your data file (CSV or Excel)
    
    Returns:
        DataFrame with your data
    """
    
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format")
    
    return data

def predict_with_your_data(file_path):
    """
    Make predictions using your own data
    
    Args:
        file_path: Path to your data file
    
    Returns:
        Predictions in Celsius
    """
    
    # Load model and scaler
    model, scaler, device = load_model_and_scaler()
    
    # Load your data
    your_data = load_your_data(file_path)
    print(f"Loaded data shape: {your_data.shape}")
    
    # Make predictions
    predictions = predict_temperature(model, scaler, your_data, device)
    
    # Convert to Celsius (assuming you have temperature column)
    if 'temperature' in your_data.columns:
        predictions_celsius = inverse_transform_temperature(
            predictions, 
            your_data['temperature'].values
        )
    else:
        predictions_celsius = predictions  # Keep scaled if no temperature reference
    
    return predictions_celsius

# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Run main demo
    predictions = main()
    
    print("\n" + "="*50)
    print("🔧 To use with your own data:")
    print("1. predictions = predict_with_your_data('your_data.csv')")
    print("2. Make sure your data has columns: temperature, relative_humidity, pressure_msl, rain")
    print("3. Optionally include 'time' column for temporal features")
    print("4. Need at least 168 hours (7 days) of data for prediction")
