import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import load_data, load_reference_data, preprocess_pipeline
from features import feature_engineering_pipeline
from model import AnomalyDetector

def test_pipeline():
    print("Testing Pipeline...")
    
    # Load data
    budget_path = os.path.join("data", "budget_sample.csv")
    ref_path = os.path.join("data", "reference_prices.csv")
    
    if not os.path.exists(budget_path) or not os.path.exists(ref_path):
        print("Data files not found.")
        return

    budget_df = load_data(budget_path)
    ref_df = load_reference_data(ref_path)
    
    print(f"Loaded {len(budget_df)} budget items and {len(ref_df)} reference items.")
    
    # 1. Preprocessing
    print("Running Preprocessing...")
    merged_df = preprocess_pipeline(budget_df, ref_df)
    print("Columns after preprocessing:", merged_df.columns)
    
    # 2. Features
    print("Running Feature Engineering...")
    featured_df = feature_engineering_pipeline(merged_df)
    print("Columns after features:", featured_df.columns)
    
    # 3. Model
    print("Running Anomaly Detection...")
    detector = AnomalyDetector(contamination=0.1)
    results_df = detector.train_predict(featured_df)
    
    print("Results:")
    print(results_df[['item_description', 'anomaly_flag', 'risk_score', 'explanation']].head(10))
    
    anomalies = results_df[results_df['anomaly_flag'] == True]
    print(f"\nFound {len(anomalies)} anomalies.")
    if len(anomalies) > 0:
        print(anomalies[['item_description', 'unit_price', 'reference_unit_price', 'explanation']])

if __name__ == "__main__":
    test_pipeline()
