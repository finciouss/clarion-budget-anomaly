# Public Sector Budget Anomaly Detection Prototype

## 📌 Project Overview
This project is a **decision-support prototype** designed to assist auditors and budget analysts in identifying potentially inefficient or abnormal line items in public sector budget documents.

Using an unsupervised machine learning approach (**Isolation Forest**), the system flags outliers based on:
1.  **Unit Price Deviations**: Comparison against a reference price database.
2.  **Structural Inconsistencies**: Discrepancies between quantity, unit price, and total price.
3.  **Statistical Outliers**: Unusual quantities or price distributions.

**⚠️ Disclaimer**: This is a research prototype, not a fraud detection system. Anomalies indicate items requiring review, not necessarily wrongdoing.

## 🚀 Features
-   **Automated Reference Matching**: Fuzzy matching of budget items to a standardized price list.
-   **Unsupervised Anomaly Detection**: Uses Isolation Forest to identify outliers without labeled data.
-   **Explainable Risk Scoring**: Provides human-readable reasons for every flag (e.g., "Price 3x higher than reference").
-   **Interactive UI**: Streamlit-based interface for easy data upload and analysis.

## 🛠️ Setup & Installation

### Prerequisites
-   Python 3.8+
-   `pip`

### Installation
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application
To launch the Streamlit dashboard:
```bash
streamlit run app.py
```

## 📂 Project Structure
-   `app.py`: Main Streamlit application entry point.
-   `src/`: Source code modules.
    -   `preprocessing.py`: Data cleaning and fuzzy matching.
    -   `features.py`: Feature engineering pipeline.
    -   `model.py`: Isolation Forest and risk scoring logic.
-   `data/`: Sample datasets.
    -   `budget_sample.csv`: Example input budget.
    -   `reference_prices.csv`: Database of standard prices.

## 📊 Methodology

### 1. Data Preprocessing
-   **Normalization**: Text is lowercased and stripped of special characters.
-   **Matching**: Budget items are matched to reference items using fuzzy string matching (Levenshtein distance). Items with low match confidence are marked but still processed if possible, or excluded from price comparison.

### 2. Feature Engineering
Key features used for detection:
-   `price_deviation_ratio`: Ratio of Unit Price to Reference Price.
-   `total_discrepancy_ratio`: Validity check of (Quantity × Unit Price) vs Total Price.
-   `quantity_z_score`: Statistical deviation of quantity.
-   `match_score`: Confidence of the reference match.

### 3. Anomaly Detection
-   **Algorithm**: Isolation Forest.
-   **Rationale**: Effective for high-dimensional datasets and detecting anomalies without ground truth labels. It isolates observations by randomly selecting a feature and then randomly selecting a split value. Anomalies are susceptible to isolation (shorter path lengths in trees).

## ⚠️ Limitations
-   **Reference Data**: Quality of results depends heavily on the completeness and accuracy of `reference_prices.csv`.
-   **False Positives**: High variability in legitimate item specifications (e.g., "Laptop") can lead to flags if the reference is too generic.
-   **No Ground Truth**: The model is unsupervised; "accuracy" is qualitative based on explainability.
