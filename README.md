# Public-Sector Budget Anomaly Detection Prototype

## Objective
- Decision-support prototype to flag potentially inefficient or abnormal budget line items in structured public-sector data.
- Scope is intentionally limited: interpretable features, one unsupervised method (Isolation Forest), human-in-the-loop review.

## Scope & Constraints
- Not a platform or production system; no deep learning; no national-scale claims.
- Uses a single unsupervised detector (Isolation Forest).
- Prioritizes interpretability over accuracy.
- Anomaly ≠ fraud. Results require human validation.

## Data Assumptions
- Input: CSV or Excel with minimal schema:
  - item_description, quantity, unit_price, total_price (optional), category (optional), year (optional), region (optional)
- Reference prices: CSV with standardized_item_name, reference_unit_price, source.
- If no reference match is found, the item is marked “Not Verified” and excluded from price-based anomaly scoring.

## Pipeline
1. Preprocess
   - Normalize text (lowercase, remove symbols, collapse whitespace).
   - Fuzzy match item_description to standardized reference names; assign match confidence (0–100).
   - Merge reference_unit_price where matched.
2. Validate
   - Check quantity × unit_price ≈ total_price (if provided); compute discrepancy ratio.
3. Features (10–15 total engineered; model uses the most informative subset)
   - price_deviation_ratio, abs_price_diff, log_price_deviation
   - total_price_discrepancy, total_discrepancy_ratio
   - log_quantity, log_unit_price
   - quantity_z_score, unit_price_cat_z_score (by category)
   - match_score
4. Detect
   - Isolation Forest trained on matched items using:
     - log_price_deviation, total_discrepancy_ratio, quantity_z_score, match_score
   - Output: anomaly_flag (True/False) and normalized risk_score (0–100).
5. Explain
   - Human-readable reasons derived from feature thresholds:
     - e.g., “Price 3.3x higher than reference”; “Total price calculation mismatch”; “Unusual quantity for category”; “Low confidence reference match”.

## Model Choice
- Isolation Forest
  - Isolates anomalies rather than modeling the normal class, aligns with “few and different” assumption.
  - Scales linearly with data size; suitable for documents.
  - Pairing with explicit feature-based explanations maintains interpretability.

## What Is an Anomaly
- Isolation Forest predicts -1 for items isolated quickly by random splits.
- In practice: large price deviations, structural inconsistencies, unusual quantities, or low-confidence matches, in combination, are more likely to be flagged.
- False positives are expected; this tool surfaces items for human review.

## Risk Scoring
- Uses the model decision_function (higher = inlier).
- Risk = MinMaxScale(-decision_function) × 100
  - 0: very likely normal
  - 100: most anomalous in the current dataset
- Each flagged row includes top contributing factors via deterministic rules.

## Output Schema
- item_description
- unit_price
- reference_unit_price
- price_deviation_ratio
- anomaly_flag (true/false)
- risk_score (0–100)
- explanation (human-readable)

## Repository Structure
- [app.py](file:///c:/Users/afini/Documents/trae_projects/clarion/app.py): Streamlit UI (upload, analyze, export).
- [src/preprocessing.py](file:///c:/Users/afini/Documents/trae_projects/clarion/src/preprocessing.py): Normalization and reference matching.
- [src/features.py](file:///c:/Users/afini/Documents/trae_projects/clarion/src/features.py): Feature engineering.
- [src/model.py](file:///c:/Users/afini/Documents/trae_projects/clarion/src/model.py): Isolation Forest, risk scoring, explanations.
- [data/reference_prices.csv](file:///c:/Users/afini/Documents/trae_projects/clarion/data/reference_prices.csv): Static reference catalog (example).
- [data/budget_sample.csv](file:///c:/Users/afini/Documents/trae_projects/clarion/data/budget_sample.csv): Example input dataset.
- [TECHNICAL_NOTE.md](file:///c:/Users/afini/Documents/trae_projects/clarion/TECHNICAL_NOTE.md): Methodology note.

## Quick Start (Local)
```bash
pip install -r requirements.txt
streamlit run app.py
```
- Upload a CSV/Excel file with the expected columns.
- Adjust contamination in the sidebar (defaults to 0.10).
- Download results as CSV from the UI.

## Deployment (Streamlit Cloud)
1. Push this repository to GitHub.
2. On share.streamlit.io, create a new app:
   - Repository: <your-username>/<repo-name>
   - Branch: main
   - App file path: app.py
3. Ensure data/reference_prices.csv is part of the repo; upload the budget file via the app.

## Limitations
- No ground-truth labels; unsupervised evaluation is case-based.
- Reference price quality and coverage directly affect usefulness.
- Anomalies are hypotheses requiring human validation; this is not fraud detection.

## Contact
- Maintainer: afini
- Purpose: research internship portfolio submission; conservative, explainable prototype.
