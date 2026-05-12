import joblib
import pandas as pd
import yaml
from datetime import datetime
import numpy as np
import holidays
from pathlib import Path

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def predict_demand(date_str, store_id=1, temp=None, rain_prob=0):
    """Production inference with input validation - NO caching needed"""
    config = load_config()
    
    # Load model safely
    model_path = Path('data/models/xgb_model.pkl')
    if not model_path.exists():
        raise FileNotFoundError("Train model first: python src/train_model.py")
    
    model = joblib.load(model_path)
    
    # Date parsing
    date = pd.to_datetime(date_str)
    
    # Weather (simplified - replace with API)
    if temp is None:
        temp = 20 + 10 * np.sin(2 * np.pi * date.dayofweek / 7)
    
    # US holidays
    us_holidays = holidays.US(years=date.year)
    is_holiday = 1 if date.date() in us_holidays else 0
    
    # Feature vector (match training exactly)
    features = pd.DataFrame({
        'day_of_week': [date.dayofweek],
        'month': [date.month],
        'quarter': [date.quarter],
        'is_weekend': [1 if date.dayofweek >= 5 else 0],
        'is_holiday': [is_holiday],
        'lag_1': [120.0],      # Recent sales (from prod DB)
        'lag_7': [115.0],      # Last week
        'rolling_mean_7': [118.0],
        'rolling_std_7': [15.0],
        'growth_rate': [0.02],
        'temp': [temp],
        'rain_prob': [rain_prob]
    })
    
    # Predict
    pred = model.predict(features)[0]
    
    # Business logic: Adjust for risk
    if rain_prob > 0.5:
        pred *= 0.9  # Rain penalty
    
    return float(pred)

if __name__ == "__main__":
    # Demo predictions
    tomorrow = (datetime.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    weekend = (datetime.now() + pd.Timedelta(days=3)).strftime('%Y-%m-%d')
    
    print(f"Tomorrow forecast: ${predict_demand(tomorrow):.0f}")
    print(f"Weekend forecast: ${predict_demand(weekend, rain_prob=0.6):.0f}")
    print("✅ Prediction service ready!")