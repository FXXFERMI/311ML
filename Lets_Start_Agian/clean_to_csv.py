import os
import pandas as pd
from data_clean import process_clean_data

base_dir = os.path.dirname(__file__)
raw_path = os.path.join(base_dir, 'test_dataset.csv')
output_path = os.path.join(base_dir, 'cleaned_test_dataset.csv')

df = process_clean_data(raw_path)

if 'Label' in df.columns:
    df = df.drop(columns=['Label'])

df.to_csv(output_path, index=False, header=False)  # No header for predict.py
print(f"✅ Cleaned test data saved to: {output_path}")
