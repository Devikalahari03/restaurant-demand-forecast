import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import LabelEncoder
import holidays
import joblib
from pathlib import Path

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_raw_data(config):
    df = pd.read_csv(config['data']['raw_path'])
    if 'date' not in df.columns:
        df['date'] = pd.to_datetime(df['Date'])  # Handle different formats
    else:
        df['date'] = pd.to_datetime(df['date'])
    if 'store_id' not in df.columns:
        df['store_id'] = 1  # Single store assumption
    return df.sort_values('date').reset_index(drop=True)

def quality_checks(df):
    issues = {}
    # Completeness
    missing = df.isnull().sum()
    issues['missing_cols'] = missing[missing > 0]
    df = df.ffill().bfill().fillna(0)
    # Validity
    df = df[df['sales'] >= 0]
    outliers = df['sales'] > df['sales'].quantile(0.99)
    df.loc[outliers, 'sales'] *= 0.8  # Winsorize
    
    return df, issues

def feature_engineer(df, config):
    us_holidays = holidays.US(years=df['date'].dt.year.unique())
    
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_holiday'] = df['date'].apply(lambda x: 1 if x.date() in us_holidays else 0)
    
    # Lags & rolls (group by store)
    df['lag_1'] = df.groupby('store_id')['sales'].shift(1)
    df['lag_7'] = df.groupby('store_id')['sales'].shift(7)
    df['rolling_mean_7'] = df.groupby('store_id')['sales'].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
    df['rolling_std_7'] = df.groupby('store_id')['sales'].rolling(window=7, min_periods=1).std().reset_index(0, drop=True)
    df['growth_rate'] = df['sales'].pct_change()
    
    # Weather simulation (replace with real API)
    df['temp'] = 20 + 10 * np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['rain_prob'] = (df['temp'] < 15).astype(float)
    
    df = df.dropna()
    return df

def split_data(df, config):
    n = int(len(df) * (1 - config['data']['test_size']))
    train = df.iloc[:n].copy()
    test = df.iloc[n:].copy()
    return train, test

if __name__ == "__main__":
    config = load_config()
    Path(config['data']['processed_path']).mkdir(parents=True, exist_ok=True)
    
    df = load_raw_data(config)
    df, issues = quality_checks(df)
    df = feature_engineer(df, config)
    train, test = split_data(df, config)
    
    train.to_csv(config['data']['processed_path'] + 'train.csv', index=False)
    test.to_csv(config['data']['processed_path'] + 'test.csv', index=False)
    
    print("✅ Pipeline complete!")
    print(f"📊 Train shape: {train.shape}, Test: {test.shape}")
    print(f"⚠️  Issues: {issues}")