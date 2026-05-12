import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import yaml

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def check_drift():
    config = load_config()
    train = pd.read_csv(config['data']['processed_path'] + 'train.csv')
    test = pd.read_csv(config['data']['processed_path'] + 'test.csv')
    
    features = ['sales', 'temp', 'day_of_week']
    drift_issues = []
    
    for feature in features:
        statistic, p_value = ks_2samp(train[feature], test[feature])
        if p_value < 0.05:
            drift_issues.append(f"{feature}: p={p_value:.4f} 🚨")
        else:
            drift_issues.append(f"{feature}: OK")
    
    return drift_issues

if __name__ == "__main__":
    issues = check_drift()
    print("🔍 Monitoring Report:")
    for issue in issues:
        print(issue)
