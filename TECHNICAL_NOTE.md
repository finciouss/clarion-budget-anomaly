# Technical Note: Anomaly Detection Methodology

## 1. Feature Selection Rationale

To build an explainable model constrained to 10-15 features, we focused on three categories of indicators:

### A. Price Validity (High Impact)
-   **`log_price_deviation`**: We use the log difference between the item's unit price and the reference price. Log transformation handles the wide range of currency values (from thousands to billions) and makes the distribution more symmetric for the model.
-   **`abs_price_diff`**: Absolute difference, useful for magnitude-based filtering.

### B. Structural Integrity (Sanity Checks)
-   **`total_discrepancy_ratio`**: $\frac{|(Quantity \times UnitPrice) - TotalPrice|}{TotalPrice}$.
    This feature catches calculation errors or data entry mistakes, which are common "structural" anomalies in budget data.

### C. Statistical Properties
-   **`quantity_z_score`**: Standardized score of quantity. Extremely large orders (e.g., 10,000 units when average is 50) are isolated.
-   **`match_score`**: The fuzzy matching score (0-100). Low scores indicate the item description deviates from standard nomenclature, which can be an anomaly in itself (obfuscation) or simply a rare item.

## 2. Model Choice: Isolation Forest

We selected **Isolation Forest** (iForest) over other methods (e.g., LOF, One-Class SVM) for the following reasons:

1.  **"Isolation" Principle**: iForest explicitly isolates anomalies rather than profiling normal points. This fits our use case where anomalies (inefficiencies) are "few and different".
2.  **Scalability**: It has linear time complexity $O(n)$, making it suitable for large budget documents.
3.  **No Distance Calculation**: Unlike LOF or k-NN, it doesn't rely on computationally expensive distance matrices, which is advantageous for high-dimensional or mixed-attribute data.
4.  **Interpretability**: While iForest is an ensemble method, we can explain its decisions by correlating the anomaly score with the input features (as implemented in the `_generate_explanation` logic).

## 3. Evaluation Approach

Since no ground-truth labels exist (unsupervised setting), we evaluate the model using a **Case-Based Validation** approach:

1.  **Synthetic Injection**: We manually inspected the `budget_sample.csv` where we intuitively know outliers (e.g., a "Luxury Office Desk" priced at 25M when reference is 2.5M).
2.  **Recall Check**: Did the model flag these known high-deviation items?
    *   *Result*: The model successfully flagged the high-deviation desk and scanner in our test run.
3.  **Explainability Check**: Does the explanation column make sense?
    *   *Result*: Flags are accompanied by reasons like "Price 10.0x higher than reference", enabling human verification.

## 4. Risk Scoring

The raw decision function of Isolation Forest yields a score where lower values indicate anomalies. We transform this into a **0-100 Risk Score**:

$$ Risk = \text{MinMaxScale}(-1 \times \text{DecisionFunction}) \times 100 $$

-   **0**: Perfectly normal (deep in the forest).
-   **100**: Most anomalous (isolated quickly).

This normalized score allows auditors to sort and prioritize their review workload.
