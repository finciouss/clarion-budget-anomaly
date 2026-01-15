import streamlit as st
import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import load_data, load_reference_data, preprocess_pipeline
from features import feature_engineering_pipeline
from model import AnomalyDetector

st.set_page_config(page_title="Budget Anomaly Detection Prototype", layout="wide")

st.title("🏛️ Public Sector Budget Anomaly Detection")
st.markdown("""
**Objective**: Prototype for flagging inefficient or abnormal budget line items based on unit price deviations and structural inconsistencies.
**Note**: This is a decision-support tool. Flags require human validation.
""")

# Sidebar
st.sidebar.header("Configuration")
contamination = st.sidebar.slider("Anomaly Contamination (Expected %)", 0.01, 0.20, 0.10)
run_button = st.sidebar.button("Run Analysis")

# Data Loading
st.subheader("1. Input Data")

uploaded_file = st.file_uploader("Upload Budget Document (CSV/Excel)", type=['csv', 'xlsx'])

# Load Reference Data (simulated static load)
REFERENCE_PATH = os.path.join("data", "reference_prices.csv")
if os.path.exists(REFERENCE_PATH):
    ref_df = load_reference_data(REFERENCE_PATH)
    st.sidebar.success(f"Loaded {len(ref_df)} reference items.")
else:
    st.error(f"Reference data not found at {REFERENCE_PATH}")
    st.stop()

if uploaded_file is not None:
    try:
        # Save uploaded file temporarily or read directly
        if uploaded_file.name.endswith('.csv'):
            budget_df = pd.read_csv(uploaded_file)
        else:
            budget_df = pd.read_excel(uploaded_file)
            
        st.write("Preview of Uploaded Data:")
        st.dataframe(budget_df.head())
        
        if run_button:
            with st.spinner("Processing data..."):
                # 1. Preprocessing
                merged_df = preprocess_pipeline(budget_df, ref_df)
                
                # 2. Feature Engineering
                featured_df = feature_engineering_pipeline(merged_df)
                
                # 3. Anomaly Detection
                detector = AnomalyDetector(contamination=contamination)
                results_df = detector.train_predict(featured_df)
                
                # Filter results for display
                output_cols = [
                    'item_description', 'unit_price', 'reference_unit_price', 
                    'price_deviation_ratio', 'total_price_discrepancy',
                    'risk_score', 'anomaly_flag', 'explanation'
                ]
                
                # Display Summary
                st.subheader("2. Analysis Results")
                
                n_anomalies = results_df['anomaly_flag'].sum()
                st.metric("Flagged Anomalies", n_anomalies, delta_color="inverse")
                
                # Split view: Anomalies vs Normal
                anomalies = results_df[results_df['anomaly_flag'] == True].sort_values('risk_score', ascending=False)
                
                st.write("### 🚩 High Risk Items (Anomalies)")
                if not anomalies.empty:
                    st.dataframe(
                        anomalies[output_cols],
                        column_config={
                            "unit_price": st.column_config.NumberColumn("unit_price", format="%,.0f"),
                            "reference_unit_price": st.column_config.NumberColumn("reference_unit_price", format="%,.0f"),
                            "price_deviation_ratio": st.column_config.NumberColumn("price_deviation_ratio", format="%.2f"),
                            "total_price_discrepancy": st.column_config.NumberColumn("total_price_discrepancy", format="%,.0f"),
                            "risk_score": st.column_config.ProgressColumn("risk_score", min_value=0, max_value=100, format="%.1f")
                        }
                    )
                else:
                    st.info("No anomalies detected based on current settings.")
                
                st.write("### All Data")
                st.dataframe(
                    results_df[output_cols],
                    column_config={
                        "unit_price": st.column_config.NumberColumn("unit_price", format="%,.0f"),
                        "reference_unit_price": st.column_config.NumberColumn("reference_unit_price", format="%,.0f"),
                        "price_deviation_ratio": st.column_config.NumberColumn("price_deviation_ratio", format="%.2f"),
                        "total_price_discrepancy": st.column_config.NumberColumn("total_price_discrepancy", format="%,.0f"),
                        "risk_score": st.column_config.ProgressColumn("risk_score", min_value=0, max_value=100, format="%.1f")
                    }
                )
                
                # Download
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Analysis Results",
                    csv,
                    "budget_anomaly_analysis.csv",
                    "text/csv",
                    key='download-csv'
                )
                
    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.exception(e)

else:
    st.info("Please upload a budget file to begin.")
    st.markdown("### Sample Data Format")
    st.code("""
item_description, quantity, unit_price, total_price, category
Laptop High Spec, 2, 45000000, 90000000, Electronics
Paper A4, 100, 50000, 5000000, Office Supplies
    """, language="csv")
