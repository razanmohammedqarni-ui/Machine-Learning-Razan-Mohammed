# Rental Price Estimation in Saudi Arabia

Extension of the ARTI 308 group project (Group 4, Section 01) on estimating residential rental prices in Saudi Arabia using the AQAR dataset from Kaggle.

## Research question

Are rental price models equally accurate across Saudi cities, and how do data quality issues (duplicates, mixed price units) affect the reported accuracy?

## Files

| File | Description |
|---|---|
| `Saudi_Arabia_Real_Estate_AQAR.csv` | AQAR rental listings (3,718 rows, 4 cities) |
| `AQAR_Week1.ipynb` | Week 1: cleaning, duplicate analysis, baseline models |

## Week 1 results

- 59% of rows are exact duplicates. Duplicate rate by city: Khobar 91.5%, Dammam 86.2%, Jeddah 53.7%, Riyadh 5.8%.
- Prices below 15,000 SAR are monthly rents and prices above 400,000 SAR are sale prices; both were removed.
- With duplicates, Random Forest reaches R² = 0.69 (MAE 11,154 SAR). After removing duplicates, R² drops to 0.42 (MAE 29,315 SAR). The difference is data leakage caused by copies of the same listing appearing in both train and test sets.

## Plan

| Week | Work |
|---|---|
| 1 | Cleaning, duplicate analysis, baseline models |
| 2 | Model comparison with cross-validation, district feature |
| 3 | Error analysis by city and property type, SHAP interpretation |
| 4 | Short report |
