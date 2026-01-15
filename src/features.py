import pandas as pd
import numpy as np

def calculate_structural_features(df):
    """
    Calculate features based on internal consistency.
    """
    df = df.copy()
    
    # 1. Total Price Discrepancy (Validation)
    # expected_total = quantity * unit_price
    # discrepancy = abs(expected_total - total_price)
    # We use a small epsilon for float comparison, but here we want the magnitude
    df['calculated_total'] = df['quantity'] * df['unit_price']
    df['total_price_discrepancy'] = (df['calculated_total'] - df['total_price']).abs()
    
    # Normalize discrepancy by total price to get a ratio (handle div by 0)
    df['total_discrepancy_ratio'] = df.apply(
        lambda row: row['total_price_discrepancy'] / row['total_price'] if row['total_price'] > 0 else 0,
        axis=1
    )
    
    # 2. Log transformations for skewed data
    df['log_quantity'] = np.log1p(df['quantity'])
    df['log_unit_price'] = np.log1p(df['unit_price'])
    
    return df

def calculate_reference_features(df):
    """
    Calculate features based on reference data.
    """
    df = df.copy()
    
    # Only calculate for matched items
    matched_mask = df['reference_unit_price'].notnull() & (df['reference_unit_price'] > 0)
    
    # 1. Price Deviation Ratio
    # ratio = unit_price / reference_unit_price
    # We center it around 1 (or 0 if we take log)
    df['price_deviation_ratio'] = np.nan
    df.loc[matched_mask, 'price_deviation_ratio'] = (
        df.loc[matched_mask, 'unit_price'] / df.loc[matched_mask, 'reference_unit_price']
    )
    
    # 2. Absolute Price Difference
    df['abs_price_diff'] = np.nan
    df.loc[matched_mask, 'abs_price_diff'] = (
        df.loc[matched_mask, 'unit_price'] - df.loc[matched_mask, 'reference_unit_price']
    ).abs()
    
    # 3. Log deviation (symmetric)
    df['log_price_deviation'] = np.nan
    df.loc[matched_mask, 'log_price_deviation'] = (
        np.log1p(df.loc[matched_mask, 'unit_price']) - np.log1p(df.loc[matched_mask, 'reference_unit_price'])
    )
    
    return df

def calculate_statistical_features(df):
    """
    Calculate statistical features like Z-scores.
    """
    df = df.copy()
    
    # Quantity Z-score (global for now, better if by category)
    if 'category' in df.columns:
        df['quantity_z_score'] = df.groupby('category')['quantity'].transform(
            lambda x: (x - x.mean()) / x.std()
        ).fillna(0)
        
        # Unit Price Z-score within category
        df['unit_price_cat_z_score'] = df.groupby('category')['unit_price'].transform(
            lambda x: (x - x.mean()) / x.std()
        ).fillna(0)
    else:
        df['quantity_z_score'] = (df['quantity'] - df['quantity'].mean()) / df['quantity'].std()
        df['unit_price_cat_z_score'] = 0 # Cannot compute
        
    return df

def feature_engineering_pipeline(df):
    """
    Master pipeline for feature engineering.
    """
    df = calculate_structural_features(df)
    df = calculate_reference_features(df)
    df = calculate_statistical_features(df)
    
    # Fill NaNs for features that might be used in the model
    # Note: We keep NaNs in 'price_deviation_ratio' to identify unmatched items later
    # but for the model input, we will filter them out.
    
    return df
