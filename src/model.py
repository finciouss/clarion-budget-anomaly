import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

class AnomalyDetector:
    def __init__(self, contamination=0.1):
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        self.feature_cols = [
            'log_price_deviation', 
            'total_discrepancy_ratio', 
            'quantity_z_score', 
            'match_score'
        ]
        self.scaler = MinMaxScaler(feature_range=(0, 100))

    def train_predict(self, df):
        """
        Train model and predict anomalies.
        Returns df with 'anomaly_flag', 'risk_score', 'explanation'.
        """
        # Filter for valid data (matched items)
        # We only score items that have a reference price match
        valid_mask = df['reference_unit_price'].notnull() & df['log_price_deviation'].notnull()
        
        if valid_mask.sum() < 5:
            # Not enough data to run anomaly detection reliably
            df['anomaly_flag'] = False
            df['risk_score'] = 0
            df['explanation'] = "Insufficient data for modeling"
            return df

        train_data = df.loc[valid_mask, self.feature_cols].fillna(0)
        
        # Fit model
        self.model.fit(train_data)
        
        # Predict (1: normal, -1: anomaly)
        preds = self.model.predict(train_data)
        scores = self.model.decision_function(train_data)
        
        # Map scores to risk (Lower score = more anomalous)
        # We want High Risk = 100, Low Risk = 0
        # decision_function yields higher values for inliers
        # So we negate it to make higher values = outliers
        neg_scores = -scores
        
        # Normalize to 0-100
        # We handle the scaling manually to ensure 0-100 range covers the dataset
        min_s = neg_scores.min()
        max_s = neg_scores.max()
        if max_s == min_s:
            risk_scores = np.zeros_like(neg_scores)
        else:
            risk_scores = (neg_scores - min_s) / (max_s - min_s) * 100
            
        # Assign back to DF
        df.loc[valid_mask, 'anomaly_flag'] = preds == -1
        df.loc[valid_mask, 'risk_score'] = risk_scores
        
        # Default for unmatched
        df.loc[~valid_mask, 'anomaly_flag'] = False
        df.loc[~valid_mask, 'risk_score'] = 0
        df.loc[~valid_mask, 'explanation'] = "Not Verified (No Reference Match)"
        
        # Generate explanations for anomalies
        df.loc[valid_mask, 'explanation'] = df.loc[valid_mask].apply(
            self._generate_explanation, axis=1
        )
        
        return df

    def _generate_explanation(self, row):
        """
        Generate human-readable explanation based on feature values.
        """
        reasons = []
        
        # Check Price Deviation
        if row['price_deviation_ratio'] > 1.5:
            reasons.append(f"Price {row['price_deviation_ratio']:.1f}x higher than reference")
        elif row['price_deviation_ratio'] < 0.5:
            reasons.append(f"Price {row['price_deviation_ratio']:.1f}x lower than reference")
            
        # Check Total Discrepancy
        if row['total_discrepancy_ratio'] > 0.05: # 5% tolerance
            reasons.append("Total price calculation mismatch")
            
        # Check Quantity
        if abs(row['quantity_z_score']) > 2:
            reasons.append("Unusual quantity for category")
            
        # Check Match Score (Low confidence match might explain deviation)
        if row['match_score'] < 80:
            reasons.append("Low confidence reference match")
            
        if not reasons:
            if row['anomaly_flag']:
                return "Statistical anomaly detected (combination of factors)"
            else:
                return "Normal"
                
        return "; ".join(reasons)

