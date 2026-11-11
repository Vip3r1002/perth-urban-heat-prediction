# Predicting Urban Heat Island Severity in Perth

Machine learning project that classifies the **Urban Heat Island (UHI) severity** (low, moderate or severe) of grid cells across the Perth metropolitan area, using urban, environmental and socio-economic features.

> Originally built for COMP5009 Data Mining at Curtin University (2026), then extended as a portfolio project.

## Why it matters
Urban heat islands make some suburbs much hotter than others, which affects health, energy use and liveability. Predicting which areas are most at risk helps councils prioritise interventions such as tree planting, cool roofs and green space.

## Data
- 5,000 labelled grid cells and 500 unlabelled cells, stored in a SQLite database
- About 30 features, including tree canopy %, surface albedo, building density and height, population density, household income, land use zone and council area
- The dataset was provided by Curtin University for coursework and is not included in this repository
- Target: `uhi_severity_class` (low 40%, moderate 42%, severe 18%, so the classes are imbalanced)

## Approach
1. **Data quality checks and cleaning**
   - Standardised inconsistent council names (14 to 7 categories), which revealed and removed 50 hidden duplicate rows
   - Replaced physically impossible values (negative albedo or building density, household size below 1) and imputed them with the median
   - Dropped a column that was 90% missing and a constant column
2. **Leakage-free pipeline** in scikit-learn: imputation, one-hot encoding and scaling, all fitted on training folds only
3. **Model selection**: stratified 5-fold cross-validation with GridSearchCV, scored by macro-F1 to handle class imbalance
4. **Evaluation**: macro-F1, accuracy, precision, recall, confusion matrix and feature importance

## Results
| Model | Macro-F1 (5-fold CV) |
|---|---|
| **Random Forest** | **0.835** |
| k-Nearest Neighbours | 0.750 |
| Naive Bayes | 0.682 |
| Decision Tree | 0.681 |

![Feature importance](images/feature_importance.png)

![Confusion matrix](images/confusion_matrix.png)

**Key insight:** The strongest drivers of heat severity were impervious surface %, tree canopy %, green infrastructure score and building density. This suggests that more greenery and fewer hard surfaces are the main levers for cooling hot suburbs.

## Tech stack
Python · SQL (SQLite) · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter

## How to run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/uhi_analysis.ipynb
```

## Repository structure
```
├── notebooks/uhi_analysis.ipynb   # full analysis
├── images/                        # charts used in this README
├── requirements.txt
└── README.md
```

## Next steps
- Try gradient boosting (XGBoost or LightGBM) and compare with Random Forest
- Use SHAP values to explain individual predictions
- Build a small Streamlit or Power BI dashboard showing high-risk areas

## Author
Built by [Vip3r1002](https://github.com/Vip3r1002). Master of Predictive Analytics student, Curtin University.
