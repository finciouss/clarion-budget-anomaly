import pandas as pd
import re
from fuzzywuzzy import process, fuzz

def normalize_text(text):
    """
    Normalize text: lowercase, remove special characters, extra spaces.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_data(file_path):
    """
    Load budget data from CSV/Excel.
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    return df

def load_reference_data(file_path):
    """
    Load reference price data.
    """
    return pd.read_csv(file_path)

def match_reference_item(item_desc, reference_items, threshold=70):
    """
    Find best match in reference items.
    Returns (match_name, score)
    """
    normalized_item = normalize_text(item_desc)
    # Get best match
    match = process.extractOne(normalized_item, reference_items, scorer=fuzz.token_sort_ratio)
    
    if match and match[1] >= threshold:
        return match[0], match[1]
    return None, 0

def preprocess_pipeline(budget_df, reference_df):
    """
    Main preprocessing pipeline.
    1. Normalize text
    2. Match with reference data
    3. Merge reference prices
    """
    # 1. Normalize
    budget_df['normalized_description'] = budget_df['item_description'].apply(normalize_text)
    reference_df['normalized_ref_name'] = reference_df['standardized_item_name'].apply(normalize_text)
    
    reference_items = reference_df['normalized_ref_name'].tolist()
    
    # 2. Match
    # This can be optimized, but using apply for prototype simplicity
    matches = budget_df['normalized_description'].apply(
        lambda x: match_reference_item(x, reference_items)
    )
    
    budget_df['matched_ref_name'] = [m[0] if m else None for m in matches]
    budget_df['match_score'] = [m[1] if m else 0 for m in matches]
    
    # 3. Merge
    # We merge on the normalized names
    merged_df = pd.merge(
        budget_df, 
        reference_df[['normalized_ref_name', 'reference_unit_price']], 
        left_on='matched_ref_name', 
        right_on='normalized_ref_name', 
        how='left'
    )
    
    # Fill NaN reference prices with 0 or handle later
    # We will mark them as "Not Verified" if no match
    
    return merged_df
