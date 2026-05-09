#!/usr/bin/env python3
"""
Enhanced Weather Forecasting LSTM Model - Inference Script
==========================================================

This script provides a standalone inference interface for the enhanced weather forecasting model.
It can load trained model weights and perform temperature predictions on new data.

Features:
- Load pre-trained model weights
- Preprocessing pipeline for input data
- Multi-variate input support (temperature, humidity, pressure, rain, temporal features)
- Real-time forecasting capabilities
- Visualization of predictions
- Export predictions to various formats

Usage:
    python inference_weather_forecaster.py --model_path best_weather_model.pth --data_path input_data.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# =============================================================================
# Model Architecture (Must match training script exactly)
# =============================================================================

class CompactForecastModel(nn.Module):
    """
    Enhanced LSTM model with CNN preprocessing for weather forecasting
    Must match the exact architecture used during training
    """
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
# Weather Forecasting Inference Class
# =============================================================================

class WeatherForecaster:
    """
    Main inference class for weather forecasting
    """
    
    def __init__(self, model_path=None, scaler_path=None, device=None):
        """
        Initialize the weather forecaster
        
        Args:
            model_path: Path to saved model weights (.pth file)
            scaler_path: Path to saved scaler (.pkl file)
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device if device else self._get_device()
        self.model = None
        self.scaler = None
        self.temp_scaler = None
        # Based on the actual trained model, only 3 features are used
        self.feature_columns = ['temperature', 'hour_of_day', 'day_of_week']
        self.input_length = 168  # 7 days
        self.output_length = 168  # 7 days
        
        # Load model and scaler if paths provided
        if model_path:
            self.load_model(model_path)
        if scaler_path:
            self.load_scaler(scaler_path)
    
    def _get_device(self):
        """Automatically detect best available device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("⚠️ Using CPU (GPU not available)")
        return device
    
    def load_model(self, model_path):
        """
        Load trained model weights
        
        Args:
            model_path: Path to saved model weights (.pth file)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Initialize model with same architecture as training
            self.model = CompactForecastModel(
                input_size=len(self.feature_columns),
                use_gru=False,
                use_cnn=True,
                dropout_rate=0.2
            ).to(self.device)
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            
            print(f"✅ Model loaded successfully from {model_path}")
            print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")
    
    def load_scaler(self, scaler_path):
        """
        Load saved data scaler
        
        Args:
            scaler_path: Path to saved scaler (.pkl file)
        """
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        try:
            self.scaler = joblib.load(scaler_path)
            print(f"✅ Scaler loaded successfully from {scaler_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading scaler: {str(e)}")
    
    def create_scaler_from_data(self, data):
        """
        Create scaler from provided data if no saved scaler available
        
        Args:
            data: DataFrame with feature columns
        """
        self.scaler = StandardScaler()
        self.scaler.fit(data[self.feature_columns])
        
        # Create temperature-only scaler for inverse transform
        self.temp_scaler = StandardScaler()
        self.temp_scaler.fit(data[['temperature']])
        
        print("✅ Scaler created from input data")
    
    def preprocess_data(self, data):
        """
        Preprocess input data for model inference
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            Preprocessed sequences ready for model input
        """
        # Ensure time column is datetime
        if 'time' in data.columns:
            data['time'] = pd.to_datetime(data['time'])
            
            # Add temporal features
            data['hour_of_day'] = data['time'].dt.hour
            data['day_of_week'] = data['time'].dt.dayofweek
        
        # Handle missing columns
        available_features = [col for col in self.feature_columns if col in data.columns]
        if len(available_features) < len(self.feature_columns):
            missing_features = set(self.feature_columns) - set(available_features)
            print(f"⚠️ Warning: Missing features: {missing_features}")
            
            # Fill missing features with default values
            for feature in missing_features:
                if feature == 'hour_of_day':
                    data[feature] = 12  # Default to noon
                elif feature == 'day_of_week':
                    data[feature] = 0  # Default to Monday
                else:
                    data[feature] = 0  # Default to 0 for other features
        
        # Select and clean data
        data = data[self.feature_columns].dropna()
        
        if len(data) < self.input_length:
            raise ValueError(f"Insufficient data: need at least {self.input_length} samples, got {len(data)}")
        
        # Create scaler if not loaded
        if self.scaler is None:
            self.create_scaler_from_data(data)
        
        # Scale data
        scaled_data = self.scaler.transform(data)
        
        return scaled_data, data
    
    def create_sequences(self, data, stride=1):
        """
        Create input sequences for model inference
        
        Args:
            data: Scaled time series data
            stride: Step size for creating sequences
            
        Returns:
            Input sequences for model
        """
        sequences = []
        for i in range(0, len(data) - self.input_length + 1, stride):
            sequences.append(data[i:i + self.input_length])
        
        return np.array(sequences)
    
    def predict(self, input_data, return_confidence=False):
        """
        Make weather predictions
        
        Args:
            input_data: Input data (DataFrame or numpy array)
            return_confidence: Whether to return prediction confidence intervals
            
        Returns:
            Predictions and optionally confidence intervals
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess data
        if isinstance(input_data, pd.DataFrame):
            scaled_data, original_data = self.preprocess_data(input_data)
        else:
            scaled_data = input_data
            original_data = None
        
        # Create sequences
        sequences = self.create_sequences(scaled_data)
        
        if len(sequences) == 0:
            raise ValueError(f"No valid sequences created from input data")
        
        # Convert to tensor
        X = torch.tensor(sequences, dtype=torch.float32).to(self.device)
        
        # Make predictions
        predictions = []
        with torch.no_grad():
            for i in range(0, len(X), 64):  # Process in batches
                batch = X[i:i + 64]
                pred = self.model(batch)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        
        # Inverse transform predictions if temperature scaler available
        if self.temp_scaler is not None:
            predictions_inv = self.temp_scaler.inverse_transform(
                predictions.reshape(-1, 1)
            ).reshape(predictions.shape)
        else:
            predictions_inv = predictions
        
        result = {
            'predictions': predictions_inv,
            'scaled_predictions': predictions,
            'input_sequences': sequences,
            'num_predictions': len(predictions_inv)
        }
        
        if return_confidence:
            # Simple confidence intervals based on prediction variance
            std = np.std(predictions_inv, axis=0)
            result['confidence_lower'] = predictions_inv - 1.96 * std
            result['confidence_upper'] = predictions_inv + 1.96 * std
        
        return result
    
    def predict_next_week(self, input_data):
        """
        Predict next week's temperature (convenience method)
        
        Args:
            input_data: Last week's data for prediction
            
        Returns:
            Next week's temperature predictions
        """
        result = self.predict(input_data)
        return result['predictions'][-1]  # Return last prediction sequence
    
    def visualize_predictions(self, predictions, actual=None, save_path=None):
        """
        Visualize predictions with optional actual values
        
        Args:
            predictions: Model predictions
            actual: Actual values (optional)
            save_path: Path to save plot (optional)
        """
        plt.figure(figsize=(15, 6))
        
        # Handle multiple prediction sequences
        if predictions.ndim == 3:
            predictions = predictions[-1]  # Take last prediction
        
        hours = range(len(predictions))
        
        plt.plot(hours, predictions, label='Predicted', color='red', linewidth=2)
        
        if actual is not None:
            if actual.ndim == 3:
                actual = actual[-1]
            plt.plot(hours, actual, label='Actual', color='blue', linewidth=2)
        
        plt.title('Weather Forecast - Temperature Prediction')
        plt.xlabel('Hours')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")
        
        plt.show()
    
    def export_predictions(self, predictions, timestamps=None, output_path='predictions.csv'):
        """
        Export predictions to CSV file
        
        Args:
            predictions: Model predictions
            timestamps: Timestamps for predictions (optional)
            output_path: Output file path
        """
        if predictions.ndim == 3:
            # Flatten multiple sequences
            predictions = predictions.reshape(-1, predictions.shape[-1])
        
        # Create DataFrame
        df = pd.DataFrame(predictions, columns=['Temperature_Prediction'])
        
        if timestamps is not None:
            df['timestamp'] = timestamps
            df = df[['timestamp', 'Temperature_Prediction']]
        
        df.to_csv(output_path, index=False)
        print(f"📁 Predictions exported to {output_path}")
    
    def save_model(self, model_path, scaler_path=None):
        """
        Save current model and scaler
        
        Args:
            model_path: Path to save model weights
            scaler_path: Path to save scaler (optional)
        """
        if self.model is None:
            raise RuntimeError("No model to save")
        
        torch.save(self.model.state_dict(), model_path)
        print(f"✅ Model saved to {model_path}")
        
        if scaler_path and self.scaler is not None:
            joblib.dump(self.scaler, scaler_path)
            print(f"✅ Scaler saved to {scaler_path}")

# =============================================================================
# Command Line Interface
# =============================================================================

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Weather Forecasting Model Inference')
    parser.add_argument('--model_path', type=str, help='Path to saved model weights (.pth)')
    parser.add_argument('--scaler_path', type=str, help='Path to saved scaler (.pkl)')
    parser.add_argument('--data_path', type=str, help='Path to input data file')
    parser.add_argument('--output_path', type=str, default='predictions.csv', help='Output file path')
    parser.add_argument('--visualize', action='store_true', help='Show prediction plots')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Initialize forecaster
    forecaster = WeatherForecaster(
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        device=args.device
    )
    
    # Load and process data
    if args.data_path:
        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f"Data file not found: {args.data_path}")
        
        # Load data based on file extension
        if args.data_path.endswith('.csv'):
            data = pd.read_csv(args.data_path)
        elif args.data_path.endswith('.xlsx'):
            data = pd.read_excel(args.data_path)
        else:
            raise ValueError("Unsupported file format. Use .csv or .xlsx")
        
        print(f"📂 Data loaded: {data.shape}")
        
        # Make predictions
        result = forecaster.predict(data, return_confidence=True)
        predictions = result['predictions']
        
        print(f"🔮 Generated {result['num_predictions']} prediction sequences")
        print(f"📊 Each sequence contains {predictions.shape[1]} hours of forecasts")
        
        # Export predictions
        forecaster.export_predictions(predictions, output_path=args.output_path)
        
        # Visualize if requested
        if args.visualize:
            forecaster.visualize_predictions(predictions)
    
    else:
        print("🚀 Weather Forecaster initialized and ready!")
        print("📋 Usage example:")
        print("  forecaster = WeatherForecaster('model.pth', 'scaler.pkl')")
        print("  predictions = forecaster.predict(your_data)")

# =============================================================================
# Example Usage
# =============================================================================

def example_usage():
    """
    Example of how to use the WeatherForecaster class
    """
    print("📚 Example Usage:")
    print("=" * 50)
    
    # Example 1: Basic usage
    print("\n1. Basic Usage:")
    print("   forecaster = WeatherForecaster('best_weather_model.pth', 'scaler.pkl')")
    print("   predictions = forecaster.predict(your_data)")
    
    # Example 2: With confidence intervals
    print("\n2. With Confidence Intervals:")
    print("   result = forecaster.predict(data, return_confidence=True)")
    print("   predictions = result['predictions']")
    print("   confidence_lower = result['confidence_lower']")
    print("   confidence_upper = result['confidence_upper']")
    
    # Example 3: Quick next week prediction
    print("\n3. Quick Next Week Prediction:")
    print("   next_week = forecaster.predict_next_week(last_week_data)")
    
    # Example 4: With visualization
    print("\n4. With Visualization:")
    print("   forecaster.visualize_predictions(predictions, actual_values)")
    
    # Example 5: Export to file
    print("\n5. Export Results:")
    print("   forecaster.export_predictions(predictions, timestamps, 'forecast.csv')")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, show example usage
        example_usage()
    else:
        # Run CLI
        main()
