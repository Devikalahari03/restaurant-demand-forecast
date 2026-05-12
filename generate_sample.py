import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Generate 5000 realistic restaurant sales rows
np.random.seed(42)
n_days = 500
dates = pd.date_range('2023-01-01', periods=n_days)
store_ids = np.random.choice([1,2,3], n_days*3)

data = []
for i, date in enumerate(dates):
    for store in [1,2,3]:
        base_sales = 200 + 50*np.sin(2*np.pi*i/7) + 30*np.random.randn()  # Weekly pattern
        promo = np.random.choice([0,1], p=[0.8, 0.2])
        sales = max(0, base_sales * (1 + 0.3*promo) + 20*np.random.randn())
        data.append({
            'date': date,
            'store_id': store,
            'sales': round(sales, 1),
            'promo': promo
        })

df = pd.DataFrame(data)
df.to_csv('data/raw/sales.csv', index=False)
print("✅ Created data/raw/sales.csv (5000 rows)")