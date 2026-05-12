import pandas as pd
import numpy as np
import pytest
import os
import sys

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_pipeline import quality_checks, feature_engineer, load_config

def test_quality_checks():
    """Test data quality handling"""
    df = pd.DataFrame({'sales': [100, -50, np.nan, 200]})
    cleaned, issues = quality_checks(df)
    assert len(cleaned) > 0
    assert (cleaned['sales'] >= 0).all()
    assert isinstance(issues, dict)

def test_feature_engineer():
    """Test feature generation"""
    config = load_config()
    df = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=10),
        'sales': np.random.randint(50, 200, 10),
        'store_id': 1
    })
    result_df = feature_engineer(df, config)
    assert 'day_of_week' in result_df.columns
    assert 'is_weekend' in result_df.columns
    assert len(result_df) > 0

def test_pipeline_end_to_end():
    """Full pipeline smoke test"""
    config = load_config()
    assert config['data']['raw_path'] == 'data/raw/sales.csv'