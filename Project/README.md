# Rental Price Estimation in Saudi Arabia

Extension of the ARTI 308 group project (Group 4, Section 01) on estimating residential rental prices in Saudi Arabia using the AQAR dataset from Kaggle.

## Research question

Are rental price models equally accurate across Saudi cities, and how do data quality issues (duplicates, mixed price units) affect the reported accuracy?

## Files

| File | Description |
|---|---|
| `Saudi_Arabia_Real_Estate_AQAR.csv` | AQAR rental listings (3,718 rows, 4 cities) |
| `Week1/AQAR_Week1.ipynb` | Week 1: cleaning, duplicate analysis, baseline models |
| `Week2/AQAR_Week2.ipynb` | Week 2: model comparison with repeated cross-validation, district target encoding, per-city errors |

## Week 1 results

- 59% of rows are exact duplicates. Duplicate rate by city: Khobar 91.5%, Dammam 86.2%, Jeddah 53.7%, Riyadh 5.8%.
- Prices below 15,000 SAR are monthly rents and prices above 400,000 SAR are sale prices; both were removed.
- With duplicates, Random Forest reaches R² = 0.69 (MAE 11,154 SAR). After removing duplicates, R² drops to 0.42 (MAE 29,315 SAR). The difference is data leakage caused by copies of the same listing appearing in both train and test sets.

## Week 2 results

Clean, deduplicated data (1,472 rows). Models trained on log(price); 5-fold cross-validation repeated 3 times (RepeatedKFold, random_state=42); MAE and R² reported on the original price scale as mean ± std over the 15 folds. Feature set A = Week 1 features (numeric columns, one-hot city and front). Feature set B = A + district, encoded inside each fold as the smoothed mean log price of the district (fallback to the city mean).

| Feature set | Model | MAE (SAR) | R² |
|---|---|---|---|
| A: Week 1 features | Linear Regression | 28,085 ± 1,778 | 0.395 ± 0.097 |
| A: Week 1 features | Ridge | 28,055 ± 1,796 | 0.396 ± 0.097 |
| A: Week 1 features | Random Forest | 27,237 ± 1,795 | 0.438 ± 0.060 |
| A: Week 1 features | Gradient Boosting | 27,836 ± 1,582 | 0.429 ± 0.063 |
| B: A + district | Linear Regression | 22,683 ± 1,592 | 0.567 ± 0.066 |
| B: A + district | Ridge | 22,636 ± 1,604 | 0.567 ± 0.065 |
| B: A + district | Random Forest | 21,999 ± 1,792 | 0.560 ± 0.050 |
| B: A + district | **Gradient Boosting** | **21,975 ± 1,561** | **0.589 ± 0.049** |

- Gradient Boosting with the district feature is the best model (MAE about 22,000 SAR, R² 0.59), with Random Forest almost tied. Within one feature set the models differ by less than the fold-to-fold standard deviation, so the features matter more than the model.
- Adding the district lowers the MAE of every model by about 5,000 to 5,500 SAR (roughly 19%) and raises R² from about 0.40–0.44 to 0.57–0.59.
- The district encoding is computed inside each cross-validation fold from the training part only; computing it on the full dataset would leak the target into the feature and inflate the scores.
- Per-city errors (pooled cross-validated predictions of the best model): Jeddah is the hardest city (MAE 30,063 SAR, MAPE 28.1%), then Khobar (MAPE 24.1%), Riyadh (20.7%) and Dammam (15.0%).
- Limitation: after deduplication Khobar has only 82 unique listings and Dammam 119, spread over about 25 districts each, so their per-city numbers are far less stable than those for Riyadh (879) and Jeddah (392).

## Plan

| Week | Work |
|---|---|
| 1 | Cleaning, duplicate analysis, baseline models |
| 2 | Model comparison with cross-validation, district feature |
| 3 | Error analysis by city and property type, SHAP interpretation |
| 4 | Short report |
