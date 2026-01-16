# Clarion — Public Sector Budget Anomaly Detection System

## 1. Project Overview
- Context: Public-sector budgets contain thousands of structured line items (goods, services, infrastructure). Manual review is slow and inconsistent, increasing the risk of inefficiencies and errors going unnoticed.
- Purpose: Clarion is a research-oriented prototype that surfaces budget line items likely to warrant human review based on price deviations and structural inconsistencies.
- Why it matters:
  - Supports governance and financial oversight by prioritizing review workload.
  - Highlights potential anomalies early without asserting wrongdoing.
- Intended users:
  - Budget analysts in ministries/agencies
  - Internal/external auditors
  - Regulators and oversight bodies

## 2. Problem Definition
- Anomaly definition:
  - A line item whose attributes (unit price, quantity, total) deviate from expected norms, reference catalogs, or internal consistency checks.
  - Examples: unit price far above/below reference; total not consistent with quantity × unit price; quantities outside typical category ranges.
- Constraints:
  - No labeled fraud/ground truth; solution must be unsupervised.
  - Interpretability is required to justify why an item is flagged.
  - Reference catalogs may be incomplete, noisy, or outdated.
- Risks of false positives:
  - Public procurement often includes legitimate variation (spec differences, vendor availability, logistics).
  - Flags are hypotheses, not findings; all results require human validation.

## 3. Data Description
- Data type: Structured budget/expenditure data.
- Key attributes:
  - item_description (string), quantity (numeric), unit_price (numeric), total_price (numeric, optional)
  - category (string, optional), year (integer, optional), region (string, optional)
  - Reference catalog: standardized_item_name, reference_unit_price, source (e.g., e-catalog, historical median)
- Additional attributes for richer analysis (not mandatory):
  - Historical spending trend, category variance, seasonality/time-based patterns, regional differences.
- Assumptions (for synthetic/simulated data):
  - Descriptions can be fuzzy-matched to a standardized reference list.
  - Quantities and prices are positive; totals may have entry errors.
  - Reference prices approximate typical/median market rates and are used for benchmarking only.

## 4. Methodology
- Rationale for unsupervised detection:
  - Lack of labeled fraud data and the need for generalizable review signals.
  - Focus on explainability rather than classification accuracy.
- Statistical methods:
  - Z-score by category for quantity and (optionally) unit price.
  - Discrepancy ratio: |(quantity × unit_price) − total_price| ÷ total_price.
  - Log transforms for heavy-tailed price/quantity distributions.
- Machine learning method:
  - Isolation Forest (single unsupervised detector)
    - Isolates anomalies via random splits; anomalies have shorter path lengths.
    - Linear time complexity, practical for large documents.
    - Outputs a decision function convertible to a 0–100 risk score.
- Trade-offs:
  - Interpretability vs detection: isolation-based score is augmented with rule-based explanations (price deviation thresholds, discrepancy flags, match confidence).
  - Coverage vs precision: unmatched items are marked “Not Verified” to avoid misleading price comparisons.

## 5. System Workflow
1. Data ingestion
   - Load CSV/Excel budget file; load static reference price catalog (CSV).
2. Preprocessing & feature engineering
   - Normalize text (lowercase, remove symbols); fuzzy-match description to standardized reference items; assign match confidence.
   - Validate internal consistency (quantity × unit_price vs total_price); compute discrepancy ratios.
   - Engineer features (≤15) including price deviation ratios, log differences, z-scores, match score.
3. Anomaly scoring
   - Train Isolation Forest on matched items using the most informative subset:
     - log_price_deviation, total_discrepancy_ratio, quantity_z_score, match_score.
   - Compute model decision scores and normalize into a 0–100 risk score.
4. Thresholding & flagging
   - Anomaly flag from model prediction (−1) with risk score ordering.
   - Rule-based explanation tags: “Price 3.3× higher than reference”, “Total price calculation mismatch”, “Unusual quantity for category”, “Low confidence reference match”.
5. Human review stage
   - Analysts audit flagged items; adjust thresholds or reference data as needed.
   - Document case outcomes to improve future evaluation.

## 6. Results & Interpretation
- Outputs:
  - For each item: anomaly_flag (true/false), risk_score (0–100), price_deviation_ratio, reference_unit_price, explanation.
  - Downloadable CSV; sortable UI table for “High Risk” items.
- Interpretation guidance:
  - Use risk_score to prioritize review; explanations indicate the main contributing factors.
  - Treat unmatched items as “Not Verified”; these are not scored on price deviation.
  - Consider context (category, region, year) before escalating.
- Limitations:
  - No ground-truth labels; results are indicative, not definitive.
  - Reference catalog coverage affects detection quality.
  - Prototype does not model procurement specifications or lifecycle context.

## 7. Explainability & Governance Considerations
- Explainability:
  - Every flagged item includes human-readable reasons derived from transparent feature thresholds.
  - Model choice (Isolation Forest) is paired with explicit rules to avoid black-box conclusions.
- Governance alignment:
  - Human-in-the-loop review ensures accountability and proportionality.
  - No claims of fraud detection or automated decision-making.
  - Documentation of assumptions, limitations, and conservative use policies.

## 8. My Role & Contributions
- I defined the problem framing and constraints (unsupervised, single method, explainable outputs).
- I designed the feature set and implemented:
  - Text normalization and fuzzy matching to a reference catalog.
  - Structural validation (discrepancy ratios) and per-category z-scores.
  - Isolation Forest scoring and 0–100 risk normalization.
  - Explanation logic mapping key features to reasons.
- I built the Streamlit UI for data upload, analysis, visualization, and CSV export.
- I prepared documentation and packaging suitable for an R&D internship portfolio submission, with conservative claims and explicit limitations.

## 9. Future Improvements
- Data enrichment:
  - Better reference catalogs, regional/temporal price indices, item specification metadata.
  - Historical baselines per category and region to contextualize deviations.
- Semi-supervised or hybrid approaches:
  - Use confirmed cases from audits to refine thresholds or weak supervision signals.
  - Combine isolation-based scores with interpretable rules for a hybrid risk model.
- Improved evaluation strategies:
  - Case-based reviews with domain experts; qualitative assessments of usefulness.
  - Tracking reviewer outcomes to calibrate flags and reduce false positives.
- Deployment considerations:
  - Secure data handling, audit logs, and role-based access in institutional settings.
  - Integration with budgeting systems for “review queues”; careful change management.
  - Continual updates to reference catalogs; governance over model updates.
