# Technical Note: Methodology and Design

## Feature Selection
We engineer a compact, interpretable set (≤15) and train the model on the most informative subset:
- Price validity:
  - `log_price_deviation`: log(unit_price) − log(reference_unit_price)
  - `abs_price_diff`: |unit_price − reference_unit_price|
  - `price_deviation_ratio`: unit_price / reference_unit_price
- Structural integrity:
  - `total_price_discrepancy`: |(quantity × unit_price) − total_price|
  - `total_discrepancy_ratio`: total_price_discrepancy / total_price
- Statistical context:
  - `log_quantity`, `log_unit_price`
  - `quantity_z_score`: per-category z-score of quantity
  - `unit_price_cat_z_score`: per-category z-score of unit price
- Matching confidence:
  - `match_score`: fuzzy match confidence to reference catalog

Model features (final subset): `log_price_deviation`, `total_discrepancy_ratio`, `quantity_z_score`, `match_score`.

## Model Choice: Isolation Forest
- Detects points that are easy to isolate via random splits; matches “few and different” assumption.
- Linear time complexity; suitable for large documents.
- Pair with deterministic rules to explain flags using feature contributions.

## Anomaly Definition
- Isolation Forest predicts -1 for anomalies.
- Final explanation draws from:
  - High price deviation (e.g., ratio > 1.5 or < 0.5)
  - High total discrepancy (>5% tolerance)
  - Extreme quantity (|z| > 2)
  - Low match confidence (< 80)

## Evaluation (Case-Based)
- No labels; evaluation is scenario-driven:
  - Inspect known high-deviation items in `data/budget_sample.csv`.
  - Confirm flags appear with clear reasons.
  - Review false positives, adjust thresholds if needed (without overfitting).

## Risk Scoring
- Use Isolation Forest decision_function:
  - Risk = MinMaxScale(− decision_function) × 100
  - 0 = most normal, 100 = most anomalous within the dataset.
- Score is sortable; reasons are provided per flagged item.

## Limitations and Guardrails
- Reference catalog may be incomplete or noisy.
- Unmatched items are “Not Verified” and excluded from price deviation scoring.
- Anomalies are hypotheses; human validation is required.
